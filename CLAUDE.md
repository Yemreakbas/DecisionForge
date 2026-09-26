# DecisionForge (formerly JEV NPC Brain)

An experimental game AI architecture using JEV as a tactical decision layer.
**Traditional AI vs JEV vs Hybrid AI**, settled in an arena rather than argued.

The project moved from `D:\Oyun Projelerim\Unity\JevNPCBrain` to
`D:\Oyun Projelerim\Unity\DecisionForge` on 2026-09-25 and continues under the
new name. Code namespaces (`JevNpcBrain.*`), `productName`, `JevCollector.exe`
and the data/model schemas (`jev-dataset/1`, `jev-model/1`) keep the old name on
purpose: renaming them buys nothing and the schemas are contracts.

Unity **6000.6.0f1**. UnityMCP is registered for this project at
`http://127.0.0.1:8080/mcp` — the Unity Editor must be open for it to respond.

This file is the project's memory. Read it before touching anything.

---

## The claim under test

> In a fair 1v1 and 3v3 arena match, a JEV-style tactical layer produces higher
> survival and fewer "dumb moments" than a hand-authored utility AI **on player
> manoeuvres nobody wrote down in advance** — at under 0.5 ms per frame.

The interesting half is the generalisation clause. Winning in the arena the
baseline was tuned on proves little; winning in a *different* arena is the finding.

## What JEV means here

JEPA-family idea: **predict in representation space, not in pixel or token space.**
Not V-JEPA itself — a ~1B-parameter video ViT cannot run per-NPC at 60 Hz, and our
world is not video anyway. Instead we train a small game-native model:

- **Encoder** — small CNN over a 32x32x7 tactical grid -> 64-128 dim latent (~500K params)
- **Predictor** — (latent, intent) -> next latent, action-conditioned
- **Planning** — sample N candidate intent sequences, roll them forward in latent
  space, score the imagined outcomes, execute the first step of the best one
- **Training** — JEPA loss against an EMA target encoder. Self-supervised, no labels,
  no reward function. Data comes free from baseline-vs-baseline matches.
- **Runtime** — ONNX via Unity Inference Engine (formerly Sentis)

## Three-layer architecture

```
REFLEX     60 Hz   hand-written, deterministic, never learned
                   aim, cover adherence, collision, flinch, animation
TACTICAL   10 Hz   the swappable layer -- UtilityBrain or JevBrain
                   emits one TacticalIntent per tick
STRATEGIC  0.2 Hz  optional, later: personality / goal weighting (LLM)
SAFETY     always  hard-rule veto over whatever the tactical layer proposed
```

## The fairness contract — do not break this

Both brains receive the **same `TacticalObservation` instance**, emit from the
**same `TacticalIntent` set**, tick at the **same 10 Hz**, and share the **same
reflex layer, weapon, movement speed, health and sensor range**.

The only difference is the function mapping observation to intent.

Corollaries that are easy to violate by accident:

- **No omniscience.** Enemy positions reach a brain only through the decayed
  `EnemyBelief` channel. Never read enemy transforms directly in a brain.
- **The baseline is not a strawman.** `UtilityBrain` has hysteresis, proper cover
  and exposure scoring, and reload-window exploitation. Weakening it to make JEV
  look good destroys the entire point of the project.
- **Mirror-match calibration.** A-vs-A and B-vs-B must land near 50%. If they do
  not, the rig is biased and no other number means anything. This gets published
  alongside the results.
- **Swapped spawns.** Every matchup is played twice with spawns exchanged.

## Arena: "Octagon"

40 x 40 m, point-symmetric. 4 full-height covers, 8 crouch-height, 2 opposing
spawns, one contested centre zone. The centre exists to force risk — without it
both teams sit in cover and never decide anything.

## Match format

1. **Captain duel (1v1)** — one captain per team. The brain itself is legible here.
2. **Squad match (3v3)** — coordination becomes legible here.

Captains get a distinct personality profile (`UtilityBrain.Weights.Captain()`), so
the duel reads as two characters rather than two copies of one function.

Weapon: hitscan, with reload and spread. Deliberately no projectile travel —
prediction should matter in the *tactical* layer, not the aiming layer.

## Metrics

| Metric | Why |
|---|---|
| Round win rate, K/D | Baseline, misleading alone |
| **Dumb moments / min** | >1.5 s in the open, back to cover, stuck. What a player actually feels |
| Exposure seconds per kill | Same result for less risk? |
| First-contact advantage | Who shoots first = who predicted better |
| Latent prediction accuracy | Is the JEV world model actually right? |
| ms / tick / NPC | It is not free; show the bill |
| **Cross-arena delta** | The real finding |

## Content layer

This project is also a piece of content, and that shapes an engineering decision:
`ITacticalBrain.LastScores` is populated **every tick, not behind a debug flag**.

The money shot is that JEV's imagination is drawable — the planner's rejected
rollouts render as fading ghost trails while the chosen one lights up. The utility
AI has no equivalent, because it only has a table of numbers. That visual contrast
*is* the architectural difference.

## Phases

1. Arena + reflex + baseline utility brains + match manager + metrics — **playable and measuring**
2. Observation logging pipeline -> `(obs, intent, next_obs)` dataset
3. Python: train the JEPA encoder/predictor, export ONNX
4. `JevBrain` + latent planner wired through Inference Engine
5. A/B measurement, then the cross-arena generalisation test

Phase 1 keeps the game playable without any ML, so the project is never hostage
to a training run.

## Conventions

- Namespaces mirror folders: `JevNpcBrain.Core`, `.Perception`, `.Tactical`, `.Arena`
- Code and docs in English; conversation with the maintainer is in Turkish
- `SelfState` field order and `ObservationChannel` order are a model contract.
  **Append only. Never insert or reorder** — it silently invalidates every checkpoint.
- Training will live in `Training/` and needs its own Python 3.12 venv
  (the machine default is 3.14, which PyTorch does not support yet)

## Git commit rules

- NEVER mention "Claude", "AI", or Anthropic in git commit messages or PR
  descriptions. That includes this file's name and phrases like "game AI" --
  say "project notes", "baseline brain".
- DO NOT add "Co-authored-by: Claude" or any AI attribution headers.
- Write conventional, standard commit messages (e.g. `feat: add user login endpoint`).

## Status

**Phase 1 is complete and running.** Scene `Assets/_Project/Scenes/01_Octagon.unity`
plays a full Utility-vs-Utility match end to end: perception -> belief -> brain ->
reflex -> weapon -> damage -> round win -> metrics -> CSV in `MatchLogs/`.

**Phase 2 is working (2026-09-25):** recording, headless collection and a
validated first training set -- see "Phase 2: the dataset" below.

**Phase 3 is working (2026-09-25):** a JEPA world model trained, evaluated on a
held-out run and exported to ONNX -- see "Phase 3: the world model" below.

**Phase 4 is working (2026-09-25):** `JevBrain` plans over that model inside
Unity, the SAFETY layer exists, and the first headless A/B is in -- see "Phase 4:
JevBrain and the first A/B" below.

**Phase 5 has its first answers (2026-09-26):** a held-out arena, "Warehouse",
and the cross-arena numbers; then a third arena, "Divide", that no model has
seen, and a world model trained on two arenas that generalises to it in play;
then procedural arenas, where 26 layouts beat two offline but not in play;
then an on-policy loop that beat UtilityStopToShoot 65.5% in Warehouse and lost
transfer to the unseen arena -- see "Phase 5", "Phase 5b", "5c" and "5d" below.

Scripts under `Assets/_Project/Scripts/`:

| Folder | Files |
|---|---|
| `Core/` | `TacticalIntent`, `SelfState`, `NpcAgent` |
| `Perception/` | `ObservationChannel`, `TacticalObservation`, `NoiseEvents`, `EnemyBelief`, `SensorGrid` |
| `Tactical/` | `ITacticalBrain`, `UtilityBrain`, `ExploringBrain` (data collection only), `JevBrain`, `JevRuntime`, `JevModelAsset` |
| `Safety/` | `ConstraintFilter` |
| `Data/` | `DatasetRecorder`, `NpzWriter` (+ `Half16`) |
| `Reflex/` | `ReflexMotor` |
| `Combat/` | `Damageable`, `Weapon` |
| `Arena/` | `ArenaLayout`, `ArenaBuilder` |
| `Match/` | `MatchDirector`, `MatchMetrics` |
| `Presentation/` | `AgentVisual`, `WeaponPose`, `WeaponModel`, `ThoughtOverlay`, `SpectatorCamera`, `AgentCameraRig`, `ArenaCameraRig`, `BroadcastCamera`, `CameraDirector`, `ActionFocus`, `ImaginationTrails` |

Editor-only: `Assets/_Project/Editor/CollectorBuild.cs` (menu **JEV > Build Data
Collector**). Python lives in `Training/` (see phases 2 and 3 below).

Scene objects: `Arena` (ArenaBuilder), `Match` (MatchDirector + ThoughtOverlay +
CameraDirector; MatchMetrics is added at runtime), `Broadcast` (ArenaCameraRig),
`Main Camera` (SpectatorCamera). Set `MatchDirector.Format` to `CaptainDuel` or
`Squad` in the Inspector; `Rounds` and `RoundTimeLimit` too.

**Dead agents stay on screen.** `MatchDirector` used to `SetActive(false)` the
victim, so the three death clips never played and a helmet cam on the victim cut
to black. Now `NpcAgent.GoLimp()` disables the CharacterController and the
ReflexMotor instead: the body plays its death clip and lies there until the next
round, but has no collider and no motor. Every consumer already filtered on
`IsAlive`, so gameplay and metrics are unchanged -- the 19-21 calibration still
describes this build. `NpcAgent.All` now includes the dead; always filter.

### Calibration result (baseline, 2026-09-24)

Utility vs Utility, captain duel, 40 rounds across two runs: **19 - 21**, zero
draws, every round decided by elimination. For n=40 at p=0.5 the standard
deviation is ~3.2, so a deviation of 1 is exactly what a fair rig looks like.
**The arena, spawn swap and reflex layer do not favour either side.** Any
difference measured from here can be attributed to the brain.

Baseline decision cost: **0.009 ms average, 1.1 ms peak.** That is the number the
JEV planner has to be weighed against, not the 0.5 ms target in the abstract.

Open finding: `center_s` came back **0.0** -- agents never enter the contested
centre at all. `ContestCenter` is scored every tick but never wins, because
`PushCoverForward` outbids it. The centre zone is currently decorative. Worth
fixing before phase 2, since a dataset where one of ten intents never fires
teaches the predictor nothing about it.

### Gotchas already paid for

- **Serialized scene values beat code defaults.** Changing a field's initialiser
  does nothing to a component already saved in a scene. This cost a wrong
  conclusion once: the code said `Rounds = 40` while the scene held 20, and the
  missing rounds were misread as unresolved matches. Check the scene, not the
  source, before drawing anything from a run.
- **Write CSVs with `CultureInfo.InvariantCulture`.** This machine is Turkish
  locale, so the default formatter emits `111,8` and splits a column. The first
  calibration CSV in `MatchLogs/archive/` is corrupt this way.
- Unity 6.3 removed `Object.GetInstanceID()`, and the `EntityId -> int` cast is
  obsolete too. Agents carry their own `NpcAgent.AgentId`, assigned by
  MatchDirector. Keep it that way: phase 2 logs observations to disk and those
  logs are only comparable across runs if ids are stable.
- `Application.runInBackground = true` is set in `MatchDirector.Start`. Without it
  the Editor throttles play mode to a couple of frames per second when unfocused,
  which looks exactly like agents that refuse to move.
- UnityMCP: `manage_components set_property` and the `scene/gameobject/{id}`
  resource both throw on these components. In edit mode,
  `manage_gameobject(action="modify", component_properties={...})` works; in play
  mode it now refuses ("cannot be used during play mode").
- `execute_code` does **not** restart play mode (re-tested 2026-09-25: twenty-odd
  calls during one running match, no reload). It is the way to drive a live match:
  set fields, call `CameraDirector.SelectKey`, kill an agent with
  `Damageable.TakeDamage`. It compiles with CodeDom (C# 6): no local functions,
  write `UnityEngine.Object` (plain `Object` is ambiguous), call extension methods
  statically, and reach project types by reflection
  (`System.Type.GetType("JevNpcBrain.Presentation.CameraDirector, Assembly-CSharp")`).
- **`Shader.Find` returns null in a player** for any shader no included asset
  references -- the editor finds everything, so this only shows up in a build.
  The first collector build died in `AgentVisual.BuildTeamRing` (URP/Unlit) before
  a round was played. Always fall back to a shader some material already carries.
- The project log is `Logs/Editor.log` (project-relative), not the usual
  `%LOCALAPPDATA%\Unity\Editor\Editor.log`. A player build blocks the editor,
  so MCP calls time out meanwhile; wait on the log's `[CollectorBuild]` line.
- New `.cs` files need `refresh_unity(scope="all")`. A scripts-only refresh
  compiled before importing them and reported the new types as missing.
- `manage_camera screenshot` with `camera="Cam_Helmet (TeamA_Captain)"` renders any
  shot, live or not. The output folder must be inside the project; use
  `Temp/...` so the images are never imported as assets.

### Art pipeline (wired, 2026-09-24)

`AgentVisual` has two modes behind one seam: assign `MatchDirector.CharacterPrefab`
and it spawns a humanoid and drives its Animator; leave it empty and it builds a
procedural mascot. No AI code knows which is in use.

**Current state: the SWAT set is imported and running.** `Assets/Swat/` holds
`Ch15_nonPBR.fbx` plus 18 clips, all converted to Humanoid (they arrived as
Generic, which silently breaks retargeting and IK). Textures were embedded and
had to be pulled out with `ModelImporter.ExtractTextures` into
`Assets/Swat/Textures/` -- before that the character rendered pure white.

Generated assets in `Assets/_Project/Art/`:
`SwatAgent.controller`, `UpperBody.mask`, `SwatAgent.prefab`, `RifleHK416.prefab`,
and the retired `RiflePlaceholder.prefab` + `.mat` (cubes, no `WeaponModel`, kept
only as a fallback asset).

**The rifle is the Gece Studio HK416** (`Assets/Gece Studio/Rifle HK416 - Free/`,
URP/Lit materials, no conversion needed). `RifleHK416.prefab` wraps the vendor
prefab without touching it: the root is the **pistol grip**, +Z down the barrel,
+Y up; the vendor model is a child rotated (0, -90, 0) -- it ships barrel along +X,
right side on -Z -- and scaled 1.85, because it ships at 0.46 m and a real HK416
is about 0.85 m. `LeftGrip` (off-hand wrist target) and `Muzzle` (barrel tip, for
VFX) are children, and a `WeaponModel` component carries the fit. Swapping rifles
means a new wrapper prefab with its own `WeaponModel`; no code changes.

Controller layout: layer 0 is a 2D Freeform Directional blend tree on
`MoveX`/`MoveY` with thresholds at the real speeds (0.34 crouch, 0.68 walk/run,
1.0 sprint), a parallel crouch tree, and a 1D death tree on `DeathDirection`.
Layer 1 is the rifle upper body under `UpperBody.mask` with Hold / Aim / Fire /
Reload. Both layers have IK Pass on, which is what `WeaponPose` needs.

Both teams wear the same model, so `AgentVisual.TintRig` blends the base colour
40% toward the team colour and drops a coloured ring at the feet -- the ring is
what carries the team read on the top-down spectator camera. Tinting runs
*before* the weapon is parented, or the rifle comes out team-coloured.

**Weapon fit: a pose in the RightHand bone's local space**, stored on the prefab's
`WeaponModel` (`HandPosition` (-0.0044, 0.0956, 0.0190), `HandEuler`
(285.66, 44.34, 58.92)). It used to be an offset in the character's frame taken
in the bind pose; that reads as intuitive and is wrong -- the hand turns ~90
degrees between the T-pose and a rifle clip, and on the aim clip the old fit put
the barrel straight up. The fit was measured in the editor against the in-game Aim composite
and checked on Fire (barrel level, off hand within 3 cm of `LeftGrip`). To retune:
play, select the rifle under the hand bone, move it, copy its local
position/rotation into `WeaponModel`. `AgentVisual.WeaponLocalPosition/Euler` are
only the fallback for a prefab without a `WeaponModel`, in the same space.

**Aim stance.** The aim and fire clips were authored with the hips bladed ~40
degrees right. The rifle layer masks them onto forward-facing locomotion, and the
upper body keeps its twist: measured, the barrel pointed **40-50 degrees left of
the target** (46 on Aim, 40 on Fire). `AgentVisual.AlignStance` fixes it by turning the rig
root (never the spine -- the Animator rewrites bones each frame and a spine turn
after IK drags the pinned hand off the rifle) until the barrel lines up with the
agent's facing. Measured each frame, only while the rifle layer is in Aim or
Fire, held through Reload, squared back up for Hold. `MoveX/MoveY` are rotated
into the turned rig's frame so the blend tree does not skate. Purely visual: the
agent root, eye and hitscan are untouched.

The off hand pins **position only** (`WeaponPose.GripRotationWeight` = 0): the
rifle clips already carry the right wrist angle, and a marker rotation even
slightly off twists it visibly. The grip releases during Reload, which takes that
hand to the magazine.

**Clips imported but not yet wired:** `Firing Rifle While Walking`,
`Reload While Walking`, `Turn 90 Left`, `Turn 90 Right`. The `Hit` parameter
exists but no state consumes it -- no hit-reaction clip was downloaded. Turn in
place also needs an angular-velocity parameter the motor does not publish yet.

**Model:** FBX, Rig > Animation Type **Humanoid** (Mixamo clips will not retarget
otherwise), ~1.8 m tall to match the CharacterController, facing +Z, pivot at the
feet, URP/Lit materials. Weapon is a separate mesh parented to the `RightHand`
bone with a `Muzzle` child for VFX -- the damage raycast still fires from the eye
transform, identically for both teams.

**Mixamo download settings:** FBX Binary, 30 fps, Keyframe Reduction `none`.
Skin: **With Skin for the character only**, `Without Skin` for every animation
clip -- otherwise each clip drags a duplicate mesh into the project. Tick
**In Place** on all locomotion clips; the CharacterController owns movement and
root motion would fight it.

**Unity import:** character FBX -> Humanoid, Create From This Model. Each clip FBX
-> Humanoid, Copy From Other Avatar (the character's). Loop Time ON for
idle/walk/run/strafe/crouch, OFF for fire/reload/vault/hit/death.

**Animator contract** (`AgentVisual` writes these every frame):

| Parameter | Type | Source |
|---|---|---|
| `MoveX`, `MoveY` | float -1..1 | `ReflexMotor.MoveLocal`, agent-local, for a 2D blend tree |
| `Speed` | float 0..1 | planar speed |
| `Crouch`, `Aiming`, `Sprinting`, `Grounded` | bool | motor state |
| `Fire` | trigger | magazine count dropping |
| `Reload` | trigger | `Weapon.IsReloading` rising edge |
| `Vault`, `Slide` | trigger | `ReflexMotor.State` transitions |
| `Hit` | trigger | `Damageable.LastDamagedAt` advancing while alive |
| `Die` + `DeathDirection` | trigger + float | death, signed by where the shot came from |

Leaning out of cover is **not** a clip -- it is a spine rotation applied in
`LateUpdate` after the Animator poses the skeleton, so it composes with any
locomotion and cannot drift out of sync with aim.

**Unarmed clips, armed character.** Mixamo has no rifle variants of jump, vault,
slide or dive, and we do not need them. The rifle is parented to the `RightHand`
bone so it survives any clip, a rifle upper-body layer (Avatar Mask: spine, arms,
hands) is layered over the borrowed clip, and `WeaponPose` IK-pins the off hand to
the weapon's `LeftGrip` child. `AgentVisual.DriveRifleLayer` blends that layer's
weight by movement state: 1.0 normally, 0.8 sprinting, 0.7 sliding, 0.25
vaulting, 0 dead. (Sprint was 0.55 until the helmet cams existed: at that weight
the unarmed sprint clip pumps the off hand across the lens at every round start.) Dropping the off hand during a vault is not a compromise -- it
is what a real operator does, one hand on the rifle and one on the wall.

Setup this needs: **IK Pass ticked** on the Animator layer driving the body, the
rifle layer at index 1 (or set `RifleLayerIndex`), rifle-layer states named
`Hold` / `Aim` / `Fire` / `Reload` (AlignStance keys off those names), and a
`WeaponModel` on the weapon prefab. The fit lives in the prefab, not in code.

**Reflex layer capabilities** (all shared by both brains, so the fairness contract
holds): walk/strafe/back, sprint when disengaged and far from the destination,
crouch, three-probe obstacle avoidance and stuck recovery.

**Vault and slide are built but switched off**
(`MatchDirector.EnableAdvancedMovement`, default false). The behaviours work and
are tested; they stay off because there are no clips for them yet and an
unanimated vault looks like a bug. Keeping them off also means the 19-21
calibration still describes the current build, since it was measured before they
existed. Turn them on and re-run calibration in the same commit as the clips.

**Clip set actually needed** (all have rifle versions on Mixamo): idle, walk
forward/back/left/right, run, sprint, turn left/right, crouch idle, crouch walk
forward/left/right, aim idle, fire, reload, hit reaction, death. Jump, vault,
slide and dive are deliberately out of scope for now.

### Cameras (content rig, 2026-09-25)

Every shot registers with `CameraDirector` (on `Match`), which owns what is on
screen. Off-air cameras have their `Camera` component disabled, never their
GameObject, so each rig keeps tracking and a cut never lands on a stale frame.

| Key | Shot | Built by |
|---|---|---|
| 1 | Tactical: the top-down `Main Camera` | scene |
| 2 | Duel: side-on two-shot of the captains. Stays on its side of the line; cuts instead of gliding when its mount jumps (respawn, side flip) | `ArenaCameraRig` |
| 3 | Wide: high over the east sideline, frames everyone | `ArenaCameraRig` |
| 4 | Towers NE/NW/SW/SE: long lens on the nearest fighter. Press again to step | `ArenaCameraRig` |
| 5 | Orbit: slow circle, for round openers | `ArenaCameraRig` |
| 6 / 7 | Helmet cam, team A / B captain | `AgentCameraRig` |
| 8 / 9 | Over the shoulder, team A / B captain | `AgentCameraRig` |

Also Tab / Shift+Tab to cycle, **V for Versus** (A's helmet cam left, B's right,
tactical inset top centre -- the core content frame), P inset, O auto-director,
H hides the HUD for clean footage. Keys belong to kinds of shot and are worked
out on demand, so they never depend on spawn order. Versus and the inset can also
be ticked in the Inspector mid-match.

Helmet cam: the mount rides the head bone, but the rotation is 35% head and 65%
the agent's facing -- a gimbal, because the aim clips press the cheek to the
stock and tip the head. Pitch is clamped to 20 down / 15 up while alive. On death
the gimbal lets go and the lens falls with the body. All agent cameras live at
the scene root and are driven in world space: parented to the agent they would
inherit its 540 deg/s turns instantly. `AgentCameraRig` runs at execution order
100 so it reads the head after `AgentVisual` has turned the rig.

Auto-director, timed in real seconds rather than simulation time: Opening (first
3 s of a round) favours Orbit and Wide; Contact (anyone has line of sight)
favours Duel and the helmet or shoulder of whoever sees; Quiet favours Tactical,
Wide and the best tower. A kill cuts straight to the **killer's helmet cam** and
holds through the respawn. Minimum hold 3.5 s, a cut for variety after 8 s.
Towers are only picked when their fighter is not hidden behind cover.

**Content caveat:** before phase 4 (2026-09-25) a team set to `BrainKind.Jev`
silently played the utility baseline while the HUD and the CSV said "Jev".
Footage and CSVs from before then are Utility vs Utility and have to be labelled
that way. Since phase 4 a Jev team without a `JevModel` throws instead.

### Phase 2: the dataset (started 2026-09-25)

**Recording.** Tick `MatchDirector.CollectDataset` (or pass `-collect`) and a
`DatasetRecorder` is added at match start. It listens to `NpcAgent.Decided` -- a
static event raised after every tactical decision -- and to the director's
`RoundStarted` / `RoundEnded` / `AgentKilled`. An **episode** is one agent's life
in one round, stored contiguously, so a transition `(obs_t, intent_t, obs_t+1)`
is steps `t, t+1` of one episode and multi-step windows come free. Ticks between
rounds are dropped, and every episode closes before the respawn teleport.

**Format, schema `jev-dataset/1`.** A run folder `Datasets/<format>_<A>-vs-<B>_<time>_seed<n>/`
holds `run.json` (channel, self-field and intent names in model order, tick,
step, seed, exploration, totals; rewritten after each shard) and
`shard_NNNN.npz`. Per step: `grid` float16 (N, 7, 32, 32), `self` float32 (N, 24),
`intent` (executed), `policy_intent` (what the baseline wanted), `explored`,
`time`, and `pos_xz` -- ground truth for plots and probes only, **never a model
input**. Per episode: `ep_start/length/agent/team/captain/round/end/won`.
The grid is float16, not bytes, because EnemyBelief, AllyPresence and NoiseHeat
accumulate past 1 (NoiseHeat reaches ~10 under sustained fire). `Half16` in
`NpzWriter.cs` is a managed float->half converter, checked bit-exact against
numpy on normals, subnormals, rounding, overflow and signs.

**Exploration.** The baseline alone never picks ContestCenter and rarely flanks,
and a world model cannot learn consequences it never saw. Collection runs wrap
every agent's brain in `ExploringBrain`: with chance `ExplorationRate` (0.012)
per decision it commits to a random intent for 0.8-3 s. Both teams alike,
collection only -- the CSV of a collection run is tagged `_collect` and its
brains are labelled `Utility+explore`, so it is never read as an evaluation.
This covers the dataset gap without touching the control group, so the baseline
and its 19-21 calibration stay as they are.

**Fixed step.** Collection sets `Time.captureDeltaTime = 1/60` and timeScale 1:
the sim advances exactly 1/60 s per frame, as fast as the machine allows, and
every transition is exactly one 0.1 s tick. On a scaled variable step the
tactical tick runs at most once per frame, so it silently falls below 10 Hz of
game time whenever a frame covers more than 0.1 s. `MatchDirector.OnDestroy`
resets the step so the editor does not stay in fixed time.

**Running it.**
- Editor: tick `CollectDataset` on `Match`, set `Rounds`, play. Runs about 2.5x
  real time with rendering on.
- Headless: **JEV > Build Data Collector** once (writes `Builds/Collector/`, ~3
  min the first time, seconds after), then `.\Training\collect.ps1 -Instances 4
  -Rounds 500`. Flags the player understands: `-collect -rounds N -seed N
  -format duel|squad -explore P -out DIR`, plus `-arena octagon|warehouse` (phase 5;
  keep other arenas' runs out of `Datasets/`'s top level, or training picks them up). A player rather than batch-mode editor,
  because the editor locks the project. Headless skips `AgentVisual` and the
  camera rigs (nothing renders, no rule reads the body) and runs ~40-50x real
  time per instance: 4 instances played 400 rounds in 19 s. An exception in a
  `-collect` run quits the player with code 1 (`QuitOnException`) rather than
  hanging, and `collect.ps1` reports non-zero exit codes. The collectors' CSVs
  land in `Builds/Collector/MatchLogs/`.
- Check a run: `Training\.venv\Scripts\python Training\inspect_dataset.py [run]`
  (newest run by default; exits 1 if a check fails).

**Headless mirror match, no exploration (2026-09-25):** team A 190 - team B 210
over 400 rounds (47.5%, z = -1.0). The headless build is as fair as the editor.
The north spawn won 54% (z ~ 1.6, not significant); the spawn swap cancels it.

**Suppression was dead until phase 2.** `NpcAgent.AddSuppression` existed but
nothing called it, so self field 19 (`Suppression01`) was a constant 0 -- the
inspector's per-field ranges exposed it. `Weapon.SuppressNearMisses` now calls
it for rounds passing within 1.5 m of an enemy without hitting. The baseline
never reads the field, so its play and its calibration are unchanged. The one
run recorded before the fix sits in `Datasets/obsolete/`, out of the loaders'
reach.

`Datasets/` layout: one folder per run (exploring), `calibration/` (explore 0,
pure baseline), `obsolete/`. `find_runs()` looks exactly one level down.

`Training/.venv` is Python 3.12 with numpy, torch 2.11 (CUDA 12.8), onnx and
onnxruntime -- see `Training/requirements.txt`.
`Training/jevdata.py` is the loader: `Run.open(folder)`, `run.shards()`,
`shard.window_starts(horizon)` for rollout windows.

**First training set (2026-09-25):** `Datasets/CaptainDuel_*_seed1001..1004`,
4 x 500 rounds with exploration, collected headless in 77 s. 4,000 episodes,
294,438 steps = 8.2 h of agent experience, ~101 MB. Every run passes
`inspect_dataset.py`. Rounds: team A 990 - team B 1007 (49.6%). Detours 17.3%
of steps. Every intent covers at least 1.5% of steps (ContestCenter 1.8%
executed vs 0.1% from the baseline alone). Suppression01 is non-zero on 59%.

### Phase 3: the world model (2026-09-25)

`Training/`: `jevmodel.py` (encoder, predictor, probes), `jepa_data.py` (runs ->
flat arrays + probe labels), `train_jepa.py`, `eval_jepa.py`, `export_onnx.py`.
Checkpoints go to `Training/checkpoints/<name>/` (`model.pt` + `report.json`),
ONNX to `Training/exports/<name>/`, logs to `Training/runs/` -- all git-ignored.

```
Training\.venv\Scripts\python Training\train_jepa.py --name NAME   # ~10 min
Training\.venv\Scripts\python Training\eval_jepa.py [checkpoint]   # newest by default
Training\.venv\Scripts\python Training\export_onnx.py [checkpoint]
Training\.venv\Scripts\python Training\train_jepa.py --refit-probes Training\checkpoints\NAME
```

**Model.** Encoder: CNN over the 7x32x32 grid (log1p on EnemyBelief, AllyPresence
and NoiseHeat, inside the graph) + MLP over the self vector -> 128-d latent,
LayerNorm without affine (666K params). Predictor: residual MLP, (z, intent
one-hot) -> z one 0.1 s tick later, same LayerNorm (411K). Probes: 128 -> 128 ->
10 sigmoid heads (18K).

**Training.** Stage 1, self-supervised: the online encoder reads obs_t, the
predictor rolls through the executed intents for H = 15 ticks (1.5 s), and every
step is regressed onto the EMA target encoder's latent of the real obs_t+k. EMA
0.99 -> 0.999, VICReg variance/covariance terms on the online latents against
collapse (latent std held at ~0.99). 12k steps x 256 windows, bf16, ~10 min on
the RTX 4060 laptop GPU; the whole train set (3.3 GB fp16) lives on the GPU.
Stage 2: probes fitted on frozen *target* latents -- the space the predictor
lands in -- so the same heads can score an imagined future. No reward anywhere in
stage 1; the probes are supervised, and that must be said when results are shown.

**Split.** Validation is run seed 1004, seen by neither stage. Training is every
other top-level run (seed101/102, 1001-1003: 230K steps).

**The action echo is masked.** `LastIntent01` at t+1 equals intent_t in 100% of
transitions, and `IntentHoldTime01` resets exactly on a switch, so the next
latent would carry the action's own label and the predictor could "predict" it
by copying. The encoder multiplies both by 0 (`jevmodel.ECHO_FIELDS`); the input
stays 24 wide, so the observation contract and Unity are untouched. An
intent-identification score from a model that sees these fields is inflated.

**Results, `jepa_v1`, held out (seed 1004, 20K windows):**

| k (ticks) | 1 | 2 | 5 | 10 | 15 |
|---|---|---|---|---|---|
| imagined MSE / copy MSE | 0.31 | 0.21 | 0.22 | 0.28 | 0.32 |

"Copy" assumes nothing changes; the rollout is 3-5x closer to the real future
across the whole 1.5 s.

Intent identification (constant-intent windows; all ten intents rolled out, the
one nearest the real future wins): accuracy 0.84, **balanced 0.60** against 0.10
chance (majority class 0.57). Strong: PushCoverForward .90, Retreat .87,
ContestCenter .84, FlankRight .79, Peek .78. Weak: Suppress .22, RegroupAlly .33
(no allies in a duel), Hold .38, PushCoverFlank .42, FlankLeft .48. The
FlankLeft/FlankRight gap is unexplained.

Probes at t+1.5 s, read off the imagined latent, the real one (ceiling) and the
latent at t (floor):

| probe | imagined | ceiling | floor |
|---|---|---|---|
| health (MAE) | 0.125 | 0.047 | 0.160 |
| exposure (MAE) | 0.134 | 0.072 | 0.173 |
| line_of_sight (AUC) | 0.930 | 1.000 | 0.900 |
| in_center (AUC) | 0.993 | 0.995 | 0.790 |
| hurt_soon, next 1 s (AUC) | 0.810 | 0.843 | 0.621 |
| death_soon, next 2 s (AUC) | 0.767 | 0.844 | 0.677 |
| round_won (AUC) | 0.649 | 0.721 | 0.640 |
| position x / z (m) | 1.25 / 1.54 | 1.06 / 1.29 | 1.75 / 3.54 |

Every probe beats the floor. round_won barely does -- an even duel is decided by
spread, so one state says little about the outcome; do not lean on it in scoring.

**Position probes are display-only.** `pos_x`/`pos_z` learn `pos_xz` as a
*target*, never an input, so the planner's rollouts can be drawn as ghost trails
(~1.3-1.5 m error at 1.5 s). No planner score may use them; `model.json` marks
them `display_only`.

**ONNX, `Training/exports/jepa_v1/`** (opset 15, 6.2 MB, every graph checked
against torch, max error 2.4e-6): `encoder`, `predictor`, `probes`, and
`imagine` -- latent (1,128) + candidate intent sequences (N,15,10) one-hot ->
probes (N,15,10), the planner's whole tick in one call. `model.json` is the
contract (channel, self-field, intent and probe names in model order); JevBrain
must compare it to its enums and refuse a mismatch. onnxruntime CPU: encoder
0.22 ms, imagine with 10 candidates 1.18 ms -- roughly 150x the baseline's
0.009 ms. The imagine graph is overhead-bound (tiny matmuls), so batching every
agent's candidates into one call should cost little more than one agent.

**Gotchas paid for in phase 3:**
- The venv was recreated in place on 2026-09-26 (`python -m venv Training/.venv`
  over the existing folder rewrites the launcher and `pyvenv.cfg` and keeps
  site-packages), so `activate` and `pip` now point at this folder.
- **Do not check out a commit older than `8055304` with the venv on disk.** That
  commit untracked `Training/.venv`; moving across it makes git overwrite the
  ignored venv from the old tree and then delete every file the old tree held
  (python.exe, pyvenv.cfg, numpy, pip). It happened once, on a branch switch and
  fast-forward. Repair: `python -m venv Training/.venv`, then
  `Training\.venv\Scripts\python -m pip install numpy` -- torch and onnx, added
  later, survive.
- torch comes from the cu128 index (see `requirements.txt`), not PyPI.
- The legacy exporter (`dynamo=False`, needed for opset 15) rejects
  `F.layer_norm(z, z.shape[-1:])` as a traced shape; pass the dim as an int.

### Phase 4: JevBrain and the first A/B (2026-09-25)

**Runtime.** `com.unity.ai.inference` 2.6.1. `Assets/_Project/Models/jepa_v1/`
holds `encoder.onnx`, `imagine.onnx`, `model.json` and `JevModel.asset`
(`JevModelAsset`: the two graphs plus the contract). `JevModelAsset.LoadContract`
compares channel, self-field and intent names with the enums and throws on any
difference. `JevRuntime` owns one encoder and one imagine worker for the whole
match. Unity's output matches torch to 2.9e-6 on a recorded observation.

**Planner (`JevBrain`).** Every tick: encode, imagine the ten intents each held
for 15 ticks, score each future as a 0.93-discounted mean of
`-1.5 death_soon - 0.75 hurt_soon - 0.25 exposure + 0.25 in_cover + 1.0 round_won
+ 0.1 in_center`, skip candidates the SAFETY layer vetoes, and keep the current
intent unless another beats it by `CommitmentBonus` (0.08; captain 0.1) -- plus
the baseline's 0.6 s minimum commitment. Weights live in `JevBrain.Weights` and
are mirrored in `Training/plan_offline.py`, which runs the same scoring on
held-out states to catch a degenerate weight set before a match is spent on it.
Offline it chose nine different intents (Hold 19%, FlankRight 19%, Suppress 15%,
...) against the baseline's 50% PushCoverForward.

**SAFETY (`Safety/ConstraintFilter`)** runs in `NpcAgent.TacticalTick` after
every brain's `Decide`, outside the timed region. Three vetoes: RegroupAlly with
no allies, Suppress with nothing to fire, PushCoverForward/ContestCenter below
25% health while exposed. Each fires on **0 of 119,151** recorded baseline
decisions, so for UtilityBrain it is a no-op (a fourth rule, no advancing while
reloading in contact, was dropped: it would have overridden 1.8% of them). A
veto falls back to the brain's best allowed alternative. `JevBrain` consults the
filter while choosing, so the post-decision override never fires for it; the
CSV's `safety_vetoes` counts overrides only.

**Ghost trails (`Presentation/ImaginationTrails`).** Added by MatchDirector when
a team is Jev, not in batch mode and `ShowImagination` is on. Chosen future:
bright team colour; rejected: thin, fading; vetoed: not drawn. Paths come from
the display-only position probes as displacement from the first imagined step,
anchored where the agent stood when it decided.

**MatchDirector.** New fields `JevModel` (assigned in the scene), `ShowImagination`,
`JevInference` (Auto = GPU compute when a graphics device exists, CPU otherwise).
New flags: `-evaluate` (fixed step like `-collect`, no recording, quits when
done, CSV tagged `_eval`), `-brainA jev|utility|stopshoot`, `-brainB ...`, and
`-spreadpenalty X` (rules ablation for both teams, CSV tagged `_spreadX`;
`evaluate.ps1 -SpreadPenalty X`). `BrainKind` gained `UtilityStopToShoot`
(appended: the enum is serialized by value). Headless CSVs
now carry `_seed<N>` so parallel instances cannot overwrite each other. **The
scene keeps both teams on Utility on purpose**: the collector build reads the
same scene, and a saved Jev team would silently turn `collect.ps1` runs into JEV
data. Pick brains with the flags or in the Inspector without saving.

**Running an A/B.** Build once (**JEV > Build Data Collector**), then
`.\Training\evaluate.ps1 -Instances 4 -Rounds 100 [-Brain jev -Against utility]`.
Half the instances put the tested brain on team A, half on B; the summary
(`Training/summarize_matches.py`) pools them with a Wilson interval and a z-score
and splits mirror matches by side. Logs: `Builds/Collector/EvalLogs/`; every
round's ending is logged as `[Round] N elimination|timeout winner=...`.

**Results, captain duel, Octagon, headless, 4 x 100 rounds each (same build):**

| Matchup | Result | Notes |
|---|---|---|
| Utility vs Utility | A 201 - B 199 (50.2%, z = +0.10), 0 draws | the SAFETY layer did not unbalance the rig |
| JEV vs JEV | A 204 - B 173 (54.1%, z = +1.60), 23 draws | inside noise (CI 49-59%) |
| **JEV vs Utility** | **JEV 306 - 85 (78.3%, 95% CI 73.9-82.1%, z = +11.2)**, 9 draws | 78.5% as team A, 78.1% as team B |

JEV vs Utility per round: K/D 292/76, first contact 262 vs 136, **dumb moments
2.27 vs 0.92**, exposure 7.9 vs 7.8 s, centre 0.0 vs 0.2 s. Decision cost on the
CPU backend in the player: **6.7 ms average** (first call ~120 ms, model
warm-up) against the baseline's 0.004 ms and the 0.5 ms target.

**Why JEV wins, as far as it is understood.** `Weapon.TryFire` spreads
`1.2 + 2.5 x moveSpeed01` degrees, so standing still is about 3x as accurate
as running. The baseline spends half its decisions walking to better cover and
fights on the move; the planner imagined that standing and shooting in contact
leads to fewer `hurt_soon`/`death_soon` futures, and it wins first contact
262-136. Nobody wrote "stop to shoot" into either brain. The same behaviour is
what the dumb-moment metric counts ("motionless, exposed, uncovered for 1.5 s"),
so JEV wins while scoring 2.5x worse on the metric meant to measure what a
player perceives. Report both halves; the claim under test is not yet
supported, because it is about *fewer* dumb moments and about a different arena.

**The controls settled it: the edge is the spread mechanic, not the planner.**
Same build, 4 x 100 rounds each. `UtilityStopToShoot` is `UtilityBrain` with one
weight changed (`ContactHoldBonus` 0.5 -> 2: hold while in contact and loaded);
`-spreadpenalty 0` sets `Weapon.MovingSpreadPenalty` to 0 for both teams.

| Matchup | Result |
|---|---|
| UtilityStopToShoot vs Utility | **79.5%** (318-82, z = +11.8) -- the one-line fix matches JEV's 78.3% |
| **JEV vs UtilityStopToShoot** | **JEV 43.0%** (169-224, CI 38.2-47.9%, z = -2.77) -- JEV loses |
| UtilityStopToShoot mirror | A 46.8% (187-213, z = -1.30) -- the stronger control is calibrated |
| JEV vs Utility, spread penalty 0 | JEV 55.5% (218-175, CI 50.5-60.3%, z = +2.17) |
| Utility mirror, spread penalty 0 | A 45.4% (181-218, z = -1.85) -- the rig's own swing under these rules |

Reading: the world model found, from data alone, the one mechanic the baseline's
author had missed -- and that is all the 78% was. A designer who knows the trick
writes one line and beats JEV head to head. With the mechanic removed, JEV's
residual 5.5 points are no larger than the mirror's own 4.6-point swing. The
claim under test is **not supported** in-arena; what JEV did prove is that a
world model is a working *discovery* tool for gaps in a hand-authored AI.
(Under the ablation rules the SAFETY layer overrode the baseline once in 400
rounds -- the "never fires on the baseline" check was made under the real rules.)

**Honest limits.** Same arena as the training data (the generalisation clause
is untested); one model, one weight set, a 1.5 s horizon and constant-intent
candidates; the ablation runs a world model trained under the real rules; the
cost is 13x over budget.

**Metric bugs found and fixed while measuring (both pre-date phase 4):**
- `RoundEnded?.Invoke(round, ResolveTimeout())` -- `?.` skips evaluating the
  arguments when nobody listens, so in a headless `-evaluate` run (no recorder,
  no cameras) timeout rounds were never counted. `-collect` and editor runs had
  listeners, so the earlier calibrations are unaffected.
- A same-frame trade (both sides wiped) counted as neither win nor draw; it is
  a draw now. The 19-21 calibration had none.

**Gotchas paid for in phase 4:**
- In the editor the CPU backend is **~240 ms per decision**: Burst refuses to
  compile Inference Engine jobs synchronously ("Synchronous compilation was
  requested ... not allowed") and they run as managed code. GPU compute in the
  editor: ~9-10 ms. A player build compiles Burst ahead of time: 6.7 ms on CPU.
- CPU tensors can be read in place (`CompleteAllPendingOperations` +
  `AsReadOnlySpan`); GPU tensors throw on that and need `DownloadToArray`.
- `execute_menu_item` for the collector build times out after 30 s while the
  build runs; wait for the `[CollectorBuild]` line in `Logs/Editor.log`.

### Phase 5: the held-out arena (2026-09-26)

**Warehouse** (`ArenaBuilder.Kind`, or `-arena warehouse` in a player;
`evaluate.ps1 -Arena warehouse`). 40 x 40 m square; two 9 m shelves and two 7 m
partitions that cut the floor into lanes; a pillar pair; a column on each spawn
diagonal; twelve crates, two of them on the centre's rim; spawns in opposite
corners. Every piece is placed as a pair through the origin -- occupancy checked
cell by cell: 0 asymmetric cells. The scene stays on Octagon. CSVs of any other
arena carry its name (`..._eval_Warehouse_...`).

Two things the first Warehouse run exposed:
- **Neither brain searches for an enemy it has never seen.** With no belief the
  threat axis is the agent's own facing and both sit in the nearest cover. From
  the true corners (44 m apart, beyond the 40 m `SightRange`) every round of a
  smoke test ran out the clock without contact. Spawns are inset to 37 m. Octagon
  never showed this because its spawns see each other across the open centre.
- **`ArenaBuilder.Stamp` marks a cell only if its centre is inside the box**, and
  centres sit at -19.375 + 1.25k. A 1.2 m wall centred between two of them stamps
  nothing -- solid to physics, invisible to line of sight and to the observation.
  Warehouse pieces are aligned to cell centres. Stamp itself was left alone:
  changing it would change Octagon's observations under the trained model.

Warehouse times out far more often (12% draws in JEV vs Utility) -- lanes make
stalemates, and the timeout rule (centre, then health) decides those rounds.

**Results, captain duel, 4 x 100 rounds each, same build:**

| Matchup | Octagon | **Warehouse** | Delta |
|---|---|---|---|
| Utility mirror | 50.2% | 49.1% (z = -0.36) | rig fair |
| StopToShoot mirror | 46.8% | 47.7% (z = -0.90) | rig fair |
| JEV vs Utility | 78.3% | **74.6%** (CI 69.8-78.9) | -3.7 |
| StopToShoot vs Utility | 79.5% | **59.1%** (CI 54.1-63.9) | **-20.4** |
| JEV vs StopToShoot | 43.0% | **33.4%** (CI 28.7-38.5, z = -6.25) | **-9.6** |

**The world model's imagination does not transfer; its reading of the present
does.** `eval_jepa.py` on a Warehouse dataset (baseline + exploration, 200 rounds,
`Datasets/warehouse/` -- one level down, so no training run picks it up):

| | Octagon (seed 1004) | Warehouse |
|---|---|---|
| imagined / copy MSE, k = 1 / 5 / 15 | 0.31 / 0.22 / 0.32 | **1.12 / 1.12 / 1.10** |
| intent id, balanced | 0.60 | 0.33 |
| probes on the *current* latent: LoS AUC, death_soon AUC, health MAE | 1.00, 0.88, 0.04 | 0.998, 0.85, 0.07 |
| probes on the *imagined* t+1.5 s latent vs copy floor | all beat the floor | most fall **below** the floor |

In Warehouse the predictor's futures are worse than assuming nothing changes.

**Reading.** Against the original baseline JEV's edge survives the new arena
(-3.7) while the designer's one-line fix loses most of its edge (-20.4): the
regularity JEV exploits (stand still to shoot) does not depend on the layout.
Head to head the fix still beats JEV, by more in Warehouse -- whatever JEV did
with its imagination in Octagon (positioning) came from dynamics that are
specific to Octagon, and in Warehouse its planner is scoring futures that are
worse than a copy of the present. The generalisation clause of the claim is
therefore **half supported**: a learned tactic transferred better than a
hand-written one, but the world model that was supposed to carry it did not.
Intransitive results (JEV > Utility, StopToShoot > JEV, StopToShoot > Utility
by less than JEV > Utility) are the norm in games; do not collapse them into one
ranking.

### Phase 5b: a third arena and a two-arena world model (2026-09-26)

**Divide** (`-arena divide`): square; a full-height wall across the middle with
a 3 m corridor at each side and a 12 m gap around the centre (its halves are
staggered by one cell -- z = 0 is a cell boundary); spawns north/south, 34.7 m
apart with a clear line through the gap. Point-symmetric, 0 asymmetric cells.
Stalemate-prone: 88 of 400 Utility-mirror rounds were draws.

**`jepa_v2`** = same architecture, trained on Octagon (seeds 101/102, 1001-1003)
plus Warehouse (2101-2104): 370K steps, a 5.3 GB grid gathered per batch from
RAM (`--gpu-data-gb`). Held out: Octagon 1004, Warehouse 2001, and all of Divide
(`Datasets/divide/`, seed 3001), which no model has seen:
`train_jepa.py --data Datasets Datasets/warehouse --val-seed 1004 2001 --name jepa_v2`.
In Unity it sits in `MatchDirector.AlternativeJevModels`; pick it with
`-jevmodel jepa_v2` (`evaluate.ps1 -JevModel jepa_v2`). CSV labels name the
model: `Jev:jepa_v1`, `Jev:jepa_v2`.

Offline, imagined/copy MSE at k = 5 and balanced intent id:

| | Octagon | Warehouse | Divide |
|---|---|---|---|
| v1 (Octagon only) | 0.22, 0.60 | 1.12, 0.33 | 1.14, 0.35 |
| v2 (Octagon + Warehouse) | 0.22, 0.66 | 0.54, 0.65 | **1.14, 0.36** |

Matches, 4 x 100 rounds each, win rate of the first brain over decided rounds:

| | Octagon | Warehouse | **Divide (unseen)** |
|---|---|---|---|
| JEV v1 vs Utility | 78.3 | 74.6 | 61.7 |
| **JEV v2 vs Utility** | -- | -- | **79.1** (CI 74.7-82.8) |
| StopToShoot vs Utility | 79.5 | 59.1 | 71.8 |
| JEV v1 vs StopToShoot | 43.0 | 33.4 | 27.8 |
| **JEV v2 vs StopToShoot** | 38.4 | **47.3** (z = -1.0, a tie) | **38.0** |
| Utility mirror | 50.2 | 49.1 | 51.6 (88 draws) |

**Reading.** One extra training arena lifted JEV in the arena that was added
(+13.9 against StopToShoot, now level with it) *and in the arena no model had
seen* (+10.2 against StopToShoot, +17.4 against Utility), at no significant
cost in Octagon (-4.6, intervals overlap). In Divide, v2 against the old
baseline matches what v1 managed in its own training arena, and beats the
designer's fix against the same baseline (79.1 vs 71.8) -- the generalisation
clause, measured the way the claim states it. The one-line fix still wins head
to head in two arenas of three.

**The offline metric missed it.** Imagined-vs-copy MSE on Divide did not move
between v1 and v2 (1.14 both), yet v2 plays 10-17 points better there. The
planner never needs an accurate latent, only the candidates' futures in the
right *order*. The metric to add is a ranking one: does the imagined ordering
of intents agree with what actually followed?

**Gotchas paid for here:**
- A dirty open scene stops `BuildPipeline.BuildPlayer` on a modal "Scene(s) Have
  Been Modified" dialog that nobody driving the editor over MCP can see; the
  build then just never starts. `CollectorBuild` now refuses to build a dirty
  scene and says so in the log.
- Collection after phase 4 runs exploratory detours through the SAFETY filter:
  1,300+ overrides per team per 200 Divide rounds (random RegroupAlly in a
  duel, Suppress while dry). The recorded intent is the executed one, so the
  data is consistent, but its exploration mix differs from phase 2's Octagon data.
- 32 GB RAM holds one two-arena training run, not a second big load next to it
  (a 1.5 GB allocation failed while jepa_v2 trained).

### Phase 5c: procedural arenas (2026-09-26)

**`ArenaKind.Procedural`** (`-arena procedural -layoutseed N`, arena name
`Proc<N>`): a seeded layout built from whole-cell pieces -- walls 4-8 x 1 cells,
2 x 2 blocks, 1 x 1 pillars (full height), 2 x 1 crates (crouch) -- every piece
with its twin through the origin. A piece is only placed if it keeps a cell of
floor around it, stays 3 m from every spawn and clear of the centre, leaves the
two captains a line of sight at spawn, and leaves every spawn reachable from
every other (4-connected flood fill). N/S or diagonal spawns, 6-11 pairs, both
from the seed. Checked on seeds 1, 2, 3, 7, 11, 42: 0 asymmetric cells, captain
line of sight every time. `collect.ps1 -Arena procedural -FirstLayoutSeed N`
gives instance i layout N + i.

`Datasets/procedural/`: layouts 101-124, 30 rounds each, 277K steps
(3.8K-26.5K per layout: some layouts fight at once, some stall).

**Score fidelity** (`eval_jepa.py`, new): the planner's own score of the
executed intents, computed from the imagined future and from the real one,
Spearman-correlated across windows. Unlike latent MSE it follows play:

| Spearman | Octagon | Warehouse | Divide |
|---|---|---|---|
| v1 | 0.905 | 0.837 | 0.823 |
| v2 | 0.889 | 0.894 | 0.835 |

Same ordering as the matches in every arena (v1 ahead in Octagon, v2 in
Warehouse and Divide), though the Divide gap is small next to the 10-point gap
in play. The decision itself ranks *candidates within one state*, which offline
data cannot score: only the executed intent's future was ever observed.

**`jepa_v3`**: trained on Octagon + Warehouse + the 24 procedural layouts (640K
steps, 16K updates, grid memory-mapped), validation seeds 1004, 2001 and 4124
(Proc124, held out). Divide still unseen. `MatchDirector.AlternativeJevModels`
holds v2 and v3; `-jevmodel jepa_v3`.

Offline on Divide it is the best of the three -- imagined/copy MSE at k = 5
0.98 (v1, v2: 1.14), intent id 0.44 (0.35, 0.36), score fidelity 0.864
(0.823, 0.835) -- and in play it is not:

| JEV vs StopToShoot | Octagon | Warehouse | Divide |
|---|---|---|---|
| v1 (1 arena) | 43.0 | 33.4 | 27.8 |
| **v2 (2 arenas)** | 38.4 | **47.3** | **38.0** |
| v3 (26 arenas) | 34.5 (41 draws) | 47.1 (**126 draws**) | 32.3 |
| JEV vs Utility, Divide | v1 61.7 | **v2 79.1** | v3 72.3 |

v3 plays more passively: a third of its Warehouse rounds ran out the clock.
Many procedural layouts are stalemate-prone (their episodes are long and
contact-poor), and a world model fed mostly on them learns that waiting is
safe -- which the planner, whose `in_center` weight is 0.1 and which has no term
for the timeout rule, then acts on. **More arenas is not automatically better;
what the arenas teach matters.** And the offline metrics, all computed on the
baseline's own trajectories, ranked v3 first: they cannot see a planner that
steers into states the data never covered, nor a within-state ranking of
candidates that were never tried.

**Gotcha: the scene is marked dirty with no change.** Twice the open scene was
flagged dirty right after a save, with the in-memory copy byte-identical to
the file (saved as a copy under Temp and diffed). A build of a dirty scene stops
on a modal dialog, so `CollectorBuild` refuses instead; save and build in the
same editor call when driving it over MCP.

**Gotcha: the Windows commit limit, not RAM.** With 10.5 GB of physical memory
free, only 5.6 GB could still be committed, so a 9 GB `np.empty` for the grid
fails. `jepa_data.load(..., memmap_path=...)` puts grids above 4 GB in a
memory-mapped .npy (`Training/cache/`, git-ignored): file-backed pages do not
count against commit, and the OS still caches them in free RAM.

### Phase 5d: the on-policy loop (2026-09-26)

`Datasets/onpolicy/`: JEV v2 (with exploration) against UtilityStopToShoot,
4 x 100 rounds in Octagon (seeds 6101-6104) and in Warehouse (6201-6204), 315K
steps. `collect.ps1` now takes `-BrainA/-BrainB/-JevModel`, and run.json's
`brains` names the model (`Jev:jepa_v2+explore`). **`jepa_v4`** = v2's data plus
this, validation seeds 1004, 2001, 6104; Divide unseen. `train_jepa.py` now
deletes its memory-mapped grid cache when it is done.

| JEV vs StopToShoot | Octagon | Warehouse | Divide (unseen) |
|---|---|---|---|
| v1 (1 arena) | 43.0 | 33.4 | 27.8 |
| **v2 (2 arenas)** | 38.4 | 47.3 | **38.0** |
| v3 (26 arenas) | 34.5 | 47.1 | 32.3 |
| **v4 (v2 + on-policy)** | 37.6 | **65.5** (CI 60.5-70.2, z = +5.97) | 26.5 |
| JEV vs Utility, Divide | v1 61.7 | v2 79.1 | v3 72.3 / v4 72.9 |

**JEV beat the stronger baseline for the first time** -- in Warehouse, where it
had played that opponent on-policy: 243-128. Offline, v4 looked like v2.
**And the specialisation cost generalisation:** in unseen Divide v4 fell to
26.5 against StopToShoot (v2 38.0, z ~ 3.5), and it gained nothing in Octagon,
where it had also played on-policy. In the open Octagon a firefight is two
stand-and-shoot brains trading shots, which leaves a planner little to plan;
Warehouse's lanes reward the positioning JEV imagines.

So the picture after five phases: **a learned planner can out-play a tuned
hand-written brain where it has learned the arena and the opponent, and cannot
yet where it has not.** The generalisation clause of the claim is the open
problem, and each recipe so far trades one side for the other.

### Next up

1. **On-policy over many arenas.** v4 shows on-policy data wins where it is
   collected; v2 shows a second arena buys transfer; v3 shows arena choice
   matters. The combination -- JEV collecting on-policy across a curated set of
   arenas, retrained each round -- is the recipe that targets both sides at once.
2. **Curate the arenas, do not just add them.** Reject stalemate-prone
   procedural layouts (e.g. measure time to first contact in a short headless
   run) or weight data by contact; then retry the many-arena model.
3. **Within-state ranking**: only matches measure it today. Replaying recorded
   states with each candidate forced for 1.5 s in a headless player would give
   counterfactual data -- to score world models offline and to train them.
4. **Decide the control group.** "The baseline is not a strawman" now points at
   `UtilityStopToShoot`: it is one weight away from the old baseline, calibrated
   (46.8% mirror) and beats both the old baseline and JEV. Every later JEV
   number should be quoted against it. The old baseline stays as the
   "what the designer first wrote" reference.
5. **Win where JEV has not practised.** It now wins where it played on-policy
   (Warehouse). Planner levers that do not depend on data: switch-at-k
   candidates, a longer horizon, a term for the timeout rule (v3's draws), a
   learned value instead of hand-picked probe weights.
6. **Search behaviour.** Neither brain looks for an enemy it has never seen; any
   arena whose spawns are out of sight of each other stalls. That is a gap in the
   shared action space (no "search" intent), not in either brain.
7. Cost: 6.7 ms per decision. Batch every JEV agent into one imagine call, a
   shorter horizon, fewer candidates, or a hand-written Burst MLP.
8. Dumb moments: the winning behaviour ("stand and shoot") is what the metric
   counts. Decide whether it should -- as it stands the metric penalises
   UtilityStopToShoot (3.1 per round in Octagon) harder than anyone.
9. Squad-format data (`-format squad`) and a retrain; RegroupAlly and
   AllyPresence are dead in duel data.
