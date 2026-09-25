# JEV NPC Brain

An experimental game AI architecture using JEV as a tactical decision layer.
**Traditional AI vs JEV vs Hybrid AI**, settled in an arena rather than argued.

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

## Status

**Phase 1 is complete and running.** Scene `Assets/_Project/Scenes/01_Octagon.unity`
plays a full Utility-vs-Utility match end to end: perception -> belief -> brain ->
reflex -> weapon -> damage -> round win -> metrics -> CSV in `MatchLogs/`.

**Phase 2 is working (2026-09-25):** recording, headless collection and a
validated first training set -- see "Phase 2: the dataset" below. Phase 3 is next.

Scripts under `Assets/_Project/Scripts/`:

| Folder | Files |
|---|---|
| `Core/` | `TacticalIntent`, `SelfState`, `NpcAgent` |
| `Perception/` | `ObservationChannel`, `TacticalObservation`, `NoiseEvents`, `EnemyBelief`, `SensorGrid` |
| `Tactical/` | `ITacticalBrain`, `UtilityBrain`, `ExploringBrain` (data collection only) |
| `Data/` | `DatasetRecorder`, `NpzWriter` (+ `Half16`) |
| `Reflex/` | `ReflexMotor` |
| `Combat/` | `Damageable`, `Weapon` |
| `Arena/` | `ArenaLayout`, `ArenaBuilder` |
| `Match/` | `MatchDirector`, `MatchMetrics` |
| `Presentation/` | `AgentVisual`, `WeaponPose`, `WeaponModel`, `ThoughtOverlay`, `SpectatorCamera`, `AgentCameraRig`, `ArenaCameraRig`, `BroadcastCamera`, `CameraDirector`, `ActionFocus` |

Editor-only: `Assets/_Project/Editor/CollectorBuild.cs` (menu **JEV > Build Data
Collector**). Python lives in `Training/` (see phase 2 below).

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

**Content caveat:** `JevBrain` does not exist yet (phase 4). A team set to
`BrainKind.Jev` plays the utility baseline -- the console warns, but the HUD and
the CSV still say "Jev". Footage from before phase 4 is Utility vs Utility and
has to be labelled that way.

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
  -format duel|squad -explore P -out DIR`. A player rather than batch-mode editor,
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

`Training/.venv` is Python 3.12 (numpy only so far; torch arrives in phase 3).
`Training/jevdata.py` is the loader: `Run.open(folder)`, `run.shards()`,
`shard.window_starts(horizon)` for rollout windows.

**First training set (2026-09-25):** `Datasets/CaptainDuel_*_seed1001..1004`,
4 x 500 rounds with exploration, collected headless in 77 s. 4,000 episodes,
294,438 steps = 8.2 h of agent experience, ~101 MB. Every run passes
`inspect_dataset.py`. Rounds: team A 990 - team B 1007 (49.6%). Detours 17.3%
of steps. Every intent covers at least 1.5% of steps (ContestCenter 1.8%
executed vs 0.1% from the baseline alone). Suppression01 is non-zero on 59%.

### Next up

1. Phase 3: JEPA encoder + action-conditioned predictor in PyTorch (install torch
   into `Training/.venv`), trained on `window_starts(horizon)` rollouts, then
   export ONNX. Open design question before the planner exists: training has no
   reward, so the planner still needs a way to score an imagined latent (e.g.
   small probes for health, exposure and line of sight trained on top of the
   frozen encoder).
2. Squad-format data (`-format squad`) once the duel model works.

Still unbuilt from phase 1's original list: nothing blocking, but the planned
`Safety/ConstraintFilter` veto layer does not exist yet -- add it before any
learned brain is allowed to drive.
