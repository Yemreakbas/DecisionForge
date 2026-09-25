using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Combat;
    using Core;
    using Reflex;

    /// <summary>
    /// The body an agent wears.
    ///
    /// Two modes, one seam. If <see cref="RigPrefab"/> is assigned it spawns a real
    /// humanoid and drives its Animator through the parameter names below. If it is
    /// null it builds a procedural mascot out of primitives and animates that by
    /// hand. Nothing else in the project knows or cares which one is in use, so a
    /// character can be dropped in at any point without touching the AI.
    ///
    /// Animator contract -- any controller wired to these names works:
    ///   float MoveX      -1..1 strafe, agent-local        (2D blend tree)
    ///   float MoveY      -1..1 forward/back, agent-local   (2D blend tree)
    ///   float Speed      0..1 planar speed magnitude
    ///   bool  Crouch
    ///   bool  Aiming
    ///   bool  Sprinting
    ///   bool  Grounded
    ///   trig  Fire
    ///   trig  Reload
    ///   trig  Vault
    ///   trig  Slide
    ///   trig  Hit
    ///   float DeathDirection  -1 shot from behind .. +1 shot from the front
    ///   trig  Die
    ///
    /// Leaning out of cover is deliberately NOT a clip. It is applied as a spine
    /// rotation in LateUpdate, after the Animator has written the pose, so it
    /// composes with whatever locomotion is playing and stays in sync with aim.
    /// </summary>
    public sealed class AgentVisual : MonoBehaviour
    {
        [Tooltip("Optional humanoid. Leave empty to use the procedural mascot.")]
        public GameObject RigPrefab;

        [Header("Weapon")]
        [Tooltip("Rifle model. Parented to the right hand bone, so it survives any clip. " +
                 "Its WeaponModel component carries the fit.")]
        public GameObject WeaponPrefab;

        [Tooltip("Fallback fit for a weapon prefab with no WeaponModel: the root's " +
                 "pose in the RightHand bone's local space.")]
        public Vector3 WeaponLocalPosition = new Vector3(-0.0044f, 0.0956f, 0.019f);

        public Vector3 WeaponLocalEuler = new Vector3(285.66f, 44.34f, 58.92f);

        [Tooltip("Fallback foregrip lookup for a weapon prefab with no WeaponModel.")]
        public string ForegripChildName = "LeftGrip";

        [Header("Rifle layer")]
        [Tooltip("Animator layer carrying the rifle upper-body pose over borrowed clips.")]
        public int RifleLayerIndex = 1;

        [Tooltip("How much rifle pose survives each move. Vault frees the off hand " +
                 "on purpose -- one hand on the gun, one on the wall.")]
        [Range(0f, 1f)] public float VaultRifleWeight = 0.25f;
        [Range(0f, 1f)] public float SlideRifleWeight = 0.7f;
        [Tooltip("Was 0.55. At that weight the unarmed sprint clip pumps the off hand " +
                 "up in front of the face -- invisible from a distance, but it fills " +
                 "the helmet cam at the start of every round.")]
        [Range(0f, 1f)] public float SprintRifleWeight = 0.8f;

        public float LeanDegrees = 7f;
        public float StepsPerMetre = 0.45f;

        [Header("Aim stance")]
        [Tooltip("Turn the body so the barrel points where the agent is facing. The " +
                 "aim and fire clips were authored with the hips bladed about 40 " +
                 "degrees off the aim line; layered over forward-facing locomotion " +
                 "they put the barrel 40-50 degrees left of the target.")]
        public bool AlignRifleToFacing = true;

        [Tooltip("How fast the stance closes the gap, per second.")]
        public float StanceTurnRate = 10f;

        [Range(0f, 90f)] public float MaxStanceDegrees = 75f;

        private static readonly int MoveXId = Animator.StringToHash("MoveX");
        private static readonly int MoveYId = Animator.StringToHash("MoveY");
        private static readonly int SpeedId = Animator.StringToHash("Speed");
        private static readonly int SprintingId = Animator.StringToHash("Sprinting");
        private static readonly int GroundedId = Animator.StringToHash("Grounded");
        private static readonly int ReloadId = Animator.StringToHash("Reload");
        private static readonly int VaultId = Animator.StringToHash("Vault");
        private static readonly int SlideId = Animator.StringToHash("Slide");
        private static readonly int HitId = Animator.StringToHash("Hit");
        private static readonly int DeathDirectionId = Animator.StringToHash("DeathDirection");
        private static readonly int CrouchId = Animator.StringToHash("Crouch");
        private static readonly int AimingId = Animator.StringToHash("Aiming");
        private static readonly int FireId = Animator.StringToHash("Fire");
        private static readonly int DieId = Animator.StringToHash("Die");

        // Rifle-layer state names, as they appear in SwatAgent.controller.
        private static readonly int AimStateId = Animator.StringToHash("Aim");
        private static readonly int FireStateId = Animator.StringToHash("Fire");
        private static readonly int ReloadStateId = Animator.StringToHash("Reload");

        private NpcAgent _agent;
        private ReflexMotor _motor;
        private Weapon _weapon;
        private Damageable _health;
        private Animator _animator;
        private Transform _spine;
        private WeaponPose _weaponPose;
        private float _rifleWeight = 1f;
        private float _gripWeight = 1f;
        private float _stanceYaw;
        private Transform _weaponRoot;

        private Transform _root;      // everything that leans and bobs
        private Transform _torso;
        private Transform _head;
        private Transform _gun;
        private Transform _legLeft;
        private Transform _legRight;
        private Transform _armLeft;
        private Transform _armRight;

        private float _stepPhase;
        private float _recoil;
        private int _lastAmmo = int.MinValue;
        private bool _deathPlayed;
        private bool _wasReloading;
        private float _lastHitAt = float.NegativeInfinity;
        private ReflexMotor.Locomotion _lastLocomotion = ReflexMotor.Locomotion.Normal;
        private float _lean;

        private void Awake()
        {
            _agent = GetComponent<NpcAgent>();
            _motor = GetComponent<ReflexMotor>();
            _weapon = GetComponentInChildren<Weapon>();
            _health = GetComponent<Damageable>();
        }

        /// <summary>The rig's Animator, once built. Null for the procedural mascot.</summary>
        public Animator Animator { get { return _animator; } }

        /// <summary>Called by MatchDirector right after the agent is constructed.</summary>
        public void Build(Material teamMaterial, bool captain)
        {
            if (RigPrefab != null)
            {
                var rig = Instantiate(RigPrefab, transform);
                rig.transform.localPosition = Vector3.zero;
                rig.transform.localRotation = Quaternion.identity;
                _animator = rig.GetComponentInChildren<Animator>();
                _root = rig.transform;

                if (_animator == null)
                {
                    Debug.LogWarning("[AgentVisual] RigPrefab has no Animator; it will not animate.");
                    return;
                }

                // Tint first: the weapon is parented under the rig a moment later,
                // and a team-coloured rifle looks like a bug rather than a uniform.
                TintRig(teamMaterial, captain);

                if (_animator.isHuman)
                {
                    _spine = _animator.GetBoneTransform(HumanBodyBones.Spine);
                    AttachWeapon();
                }
                else
                {
                    Debug.LogWarning("[AgentVisual] Rig is not Humanoid. Retargeted Mixamo clips, " +
                                     "the weapon IK and the procedural lean all need " +
                                     "Animation Type: Humanoid.");
                }

                return;
            }

            BuildMascot(teamMaterial, captain);
        }

        /// <summary>
        /// Makes a shared character model readable as two opposing teams.
        ///
        /// Both sides wear the same SWAT model, so without this a spectator -- and
        /// anyone watching the footage -- cannot tell who is who. The base colour is
        /// blended rather than replaced so the texture detail survives, and a ring
        /// at the feet carries the team read from a top-down camera where the body
        /// tint is barely visible.
        /// </summary>
        private void TintRig(Material teamMaterial, bool captain)
        {
            var team = teamMaterial.color;
            var tint = Color.Lerp(Color.white, team, 0.4f);

            var renderers = _root.GetComponentsInChildren<Renderer>();
            for (int r = 0; r < renderers.Length; r++)
            {
                var mats = renderers[r].materials;
                for (int m = 0; m < mats.Length; m++)
                {
                    if (mats[m].HasProperty("_BaseColor")) mats[m].SetColor("_BaseColor", tint);
                    else mats[m].color = tint;
                }
                renderers[r].materials = mats;
            }

            BuildTeamRing(team, teamMaterial.shader, captain);
        }

        private void BuildTeamRing(Color team, Shader teamMaterialShader, bool captain)
        {
            // A player build only contains shaders some included asset references,
            // and nothing references URP/Unlit -- Shader.Find returned null there
            // and the Material constructor threw before a single round was played.
            // The team material's own shader is in the build by construction.
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = teamMaterialShader;

            var ringMaterial = new Material(shader) { name = "TeamRing" };
            var bright = Color.Lerp(team, Color.white, captain ? 0.35f : 0f);
            if (ringMaterial.HasProperty("_BaseColor")) ringMaterial.SetColor("_BaseColor", bright);
            ringMaterial.color = bright;

            var ring = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            ring.name = "TeamRing";
            ring.transform.SetParent(transform, false);
            ring.transform.localPosition = Vector3.up * 0.03f;

            float radius = captain ? 0.62f : 0.5f;
            ring.transform.localScale = new Vector3(radius * 2f, 0.015f, radius * 2f);
            ring.GetComponent<MeshRenderer>().sharedMaterial = ringMaterial;

            var collider = ring.GetComponent<Collider>();
            if (Application.isPlaying) Destroy(collider);
            else DestroyImmediate(collider);
        }

        /// <summary>
        /// Puts the rifle in the right hand and pins the off hand to its foregrip.
        ///
        /// Attaching to the bone rather than playing a "rifle" clip is what lets
        /// borrowed empty-handed animations -- jump, vault, slide, dive, which
        /// Mixamo has no armed versions of -- carry a weapon convincingly.
        /// </summary>
        private void AttachWeapon()
        {
            if (WeaponPrefab == null) return;

            var hand = _animator.GetBoneTransform(HumanBodyBones.RightHand);
            if (hand == null)
            {
                Debug.LogWarning("[AgentVisual] No RightHand bone; weapon not attached.");
                return;
            }

            var weapon = Instantiate(WeaponPrefab, hand);
            _weaponRoot = weapon.transform;

            // The fit is a pose in the hand bone's own space. It used to be an
            // offset in the character's frame, taken in the bind pose -- which
            // reads as intuitive and is wrong: the hand swings through ninety
            // degrees between the T-pose and a rifle clip, and on the aim clip the
            // rifle came out pointing at the sky. Fitted against the aim and fire
            // clips instead, it holds for every clip that keeps a hand on the grip.
            var model = weapon.GetComponent<WeaponModel>();
            _weaponRoot.localPosition = model != null ? model.HandPosition : WeaponLocalPosition;
            _weaponRoot.localRotation = Quaternion.Euler(model != null ? model.HandEuler : WeaponLocalEuler);

            _weaponPose = _animator.gameObject.AddComponent<WeaponPose>();
            _weaponPose.LeftGrip = model != null && model.LeftGrip != null
                ? model.LeftGrip
                : FindChild(_weaponRoot, ForegripChildName);

            if (_weaponPose.LeftGrip == null)
                Debug.LogWarning($"[AgentVisual] Weapon has no '{ForegripChildName}' child. " +
                                 "The off hand will not be pinned to the foregrip.");
        }

        private static Transform FindChild(Transform root, string name)
        {
            if (root.name == name) return root;

            for (int i = 0; i < root.childCount; i++)
            {
                var found = FindChild(root.GetChild(i), name);
                if (found != null) return found;
            }

            return null;
        }

        /// <summary>
        /// A readable stand-in: torso, head, visor, two legs, two arms and a rifle.
        /// Primitives, but posed and proportioned so a viewer can tell at a glance
        /// which way it is facing and what it is doing -- which is the only thing
        /// the capsule was failing at.
        /// </summary>
        private void BuildMascot(Material teamMaterial, bool captain)
        {
            var dark = new Material(teamMaterial) { name = "AgentDark" };
            dark.color = teamMaterial.color * 0.45f;

            var visorMaterial = new Material(teamMaterial) { name = "AgentVisor" };
            visorMaterial.color = new Color(0.95f, 0.95f, 1f);

            _root = new GameObject("Visual").transform;
            _root.SetParent(transform, false);

            _torso = Part("Torso", _root, PrimitiveType.Capsule,
                new Vector3(0f, 1.15f, 0f), new Vector3(0.52f, 0.36f, 0.42f), teamMaterial);

            _head = Part("Head", _root, PrimitiveType.Sphere,
                new Vector3(0f, 1.62f, 0f), Vector3.one * 0.38f, dark);

            Part("Visor", _head, PrimitiveType.Cube,
                new Vector3(0f, 0.02f, 0.42f), new Vector3(0.62f, 0.26f, 0.22f), visorMaterial);

            _legLeft = Part("LegL", _root, PrimitiveType.Capsule,
                new Vector3(-0.17f, 0.42f, 0f), new Vector3(0.2f, 0.42f, 0.2f), dark);
            _legRight = Part("LegR", _root, PrimitiveType.Capsule,
                new Vector3(0.17f, 0.42f, 0f), new Vector3(0.2f, 0.42f, 0.2f), dark);

            // Tucked close to the torso -- set any wider and the silhouette reads
            // as a T-pose, which makes the whole thing look unrigged.
            _armLeft = Part("ArmL", _root, PrimitiveType.Capsule,
                new Vector3(-0.28f, 1.12f, 0.04f), new Vector3(0.13f, 0.32f, 0.13f), dark);
            _armRight = Part("ArmR", _root, PrimitiveType.Capsule,
                new Vector3(0.28f, 1.12f, 0.04f), new Vector3(0.13f, 0.32f, 0.13f), dark);

            _gun = Part("Rifle", _root, PrimitiveType.Cube,
                new Vector3(0.22f, 1.28f, 0.52f), new Vector3(0.11f, 0.13f, 0.82f), dark);

            if (captain)
            {
                // A crest so the captain is findable in a wide shot without a label.
                var crest = Part("Crest", _head, PrimitiveType.Cube,
                    new Vector3(0f, 0.55f, 0f), new Vector3(0.18f, 0.5f, 0.55f), visorMaterial);
                crest.localRotation = Quaternion.Euler(18f, 0f, 0f);
            }
        }

        private static Transform Part(string name, Transform parent, PrimitiveType type,
            Vector3 localPosition, Vector3 localScale, Material material)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPosition;
            go.transform.localScale = localScale;
            go.GetComponent<MeshRenderer>().sharedMaterial = material;

            var collider = go.GetComponent<Collider>();
            if (Application.isPlaying) Destroy(collider);
            else DestroyImmediate(collider);

            return go.transform;
        }

        private void LateUpdate()
        {
            if (_agent == null || _motor == null) return;

            float speed01 = _motor.Speed01;
            bool aiming = _agent.CurrentIntent == Core.TacticalIntent.Suppress
                          || _agent.Observation.Self.HasLineOfSight > 0.5f;

            DetectShot();
            DetectReload();
            DetectHit();
            DetectLocomotionChange();

            if (_animator != null)
            {
                DriveAnimator(speed01, aiming);
                DriveRifleLayer(aiming);
                AlignStance();
                ApplyLean(aiming);
                return;
            }

            AnimateMascot(speed01, aiming);
        }

        /// <summary>
        /// The weapon does not raise an event, so a drop in the magazine is how we
        /// notice a shot. Cheap, and it cannot get out of sync with the actual fire
        /// rate the way a duplicated timer would.
        /// </summary>
        private void DetectShot()
        {
            if (_weapon == null) return;

            if (_lastAmmo != int.MinValue && _weapon.Ammo < _lastAmmo)
            {
                _recoil = 1f;
                if (_animator != null) _animator.SetTrigger(FireId);
            }

            _lastAmmo = _weapon.Ammo;
            _recoil = Mathf.MoveTowards(_recoil, 0f, Time.deltaTime * 7f);
        }

        private void DriveAnimator(float speed01, bool aiming)
        {
            // The blend tree reads direction in the rig's frame, and the aim stance
            // turns the rig away from the agent's. Without this, advancing while
            // bladed would play a forward walk on a body facing sideways, and the
            // feet would skate.
            var move = _motor.MoveLocal;
            if (_stanceYaw != 0f)
            {
                var local = Quaternion.Euler(0f, -_stanceYaw, 0f) * new Vector3(move.x, 0f, move.y);
                move = new Vector2(local.x, local.z);
            }

            _animator.SetFloat(MoveXId, move.x, 0.08f, Time.deltaTime);
            _animator.SetFloat(MoveYId, move.y, 0.08f, Time.deltaTime);
            _animator.SetFloat(SpeedId, speed01, 0.08f, Time.deltaTime);

            _animator.SetBool(CrouchId, _motor.IsCrouched);
            _animator.SetBool(AimingId, aiming);
            _animator.SetBool(SprintingId, _motor.IsSprinting);
            _animator.SetBool(GroundedId, _motor.IsGrounded);

            if (_agent.IsAlive || _deathPlayed) return;

            _deathPlayed = true;

            // Which way they fall should match which way they were shot from.
            var from = _health != null ? _health.LastDamageDirection : transform.forward;
            float facing = Vector3.Dot(transform.forward, new Vector3(from.x, 0f, from.z).normalized);
            _animator.SetFloat(DeathDirectionId, facing);
            _animator.SetTrigger(DieId);
        }

        /// <summary>
        /// Blends how much rifle pose survives whatever clip is playing.
        ///
        /// Full weight while fighting, so the weapon stays shouldered. Partial
        /// during a slide or sprint, where a real operator lets the gun ride lower.
        /// Almost nothing during a vault, which frees the off hand for the wall --
        /// the borrowed empty-handed clip then reads as deliberate rather than as
        /// a missing asset.
        /// </summary>
        private void DriveRifleLayer(bool aiming)
        {
            float target;

            if (!_agent.IsAlive) target = 0f;
            else if (_motor.IsVaulting) target = VaultRifleWeight;
            else if (_motor.IsSliding) target = SlideRifleWeight;
            else if (_motor.IsSprinting) target = SprintRifleWeight;
            else target = 1f;

            _rifleWeight = Mathf.MoveTowards(_rifleWeight, target, Time.deltaTime * 4f);

            if (RifleLayerIndex > 0 && RifleLayerIndex < _animator.layerCount)
                _animator.SetLayerWeight(RifleLayerIndex, _rifleWeight);

            if (_weaponPose == null) return;

            // The off hand follows the same curve, so grip and pose never disagree
            // -- except through a reload, where the clip takes that hand to the
            // magazine and pinning it to the handguard would erase the animation.
            float grip = RifleStateIs(ReloadStateId) ? 0f : _rifleWeight;
            _gripWeight = Mathf.MoveTowards(_gripWeight, grip, Time.deltaTime * 6f);

            _weaponPose.GripWeight = _gripWeight;
            _weaponPose.LookWeight = aiming && _agent.IsAlive ? 0.6f : 0f;
        }

        /// <summary>
        /// Turns the whole rig so the barrel points where the agent is facing.
        ///
        /// The aim and fire clips were authored with the hips bladed about 40
        /// degrees off the aim line. The rifle layer masks them onto forward-facing
        /// locomotion and the upper body brings its twist along: measured in the
        /// editor, the barrel lands 40-50 degrees left of the target. Turning the
        /// rig root -- never the spine -- closes that gap without fighting the
        /// Animator, which rewrites every bone each frame but leaves its own
        /// transform alone, and without breaking the off-hand IK, which rotates
        /// rigidly along with everything else.
        ///
        /// Measured every frame rather than hard-coded, so it stays right whatever
        /// the clip set. Only while the rifle is shouldered: the low-ready hold is
        /// meant to angle across the body, and a reload keeps the stance it
        /// started in.
        /// </summary>
        private void AlignStance()
        {
            if (!AlignRifleToFacing || _weaponRoot == null || _root == null) return;
            if (RifleLayerIndex <= 0 || RifleLayerIndex >= _animator.layerCount) return;

            int current = _animator.GetCurrentAnimatorStateInfo(RifleLayerIndex).shortNameHash;
            int next = _animator.IsInTransition(RifleLayerIndex)
                ? _animator.GetNextAnimatorStateInfo(RifleLayerIndex).shortNameHash
                : current;

            float target;
            if (!_agent.IsAlive || current == ReloadStateId || next == ReloadStateId)
            {
                target = _stanceYaw;
            }
            else if (IsShouldered(current) && IsShouldered(next))
            {
                var barrel = Vector3.ProjectOnPlane(_weaponRoot.forward, Vector3.up);
                if (barrel.sqrMagnitude < 0.01f) return;
                target = _stanceYaw + Vector3.SignedAngle(barrel, transform.forward, Vector3.up);
            }
            else
            {
                // Raising the rifle: hold still until the blend settles, or the
                // measurement chases the low-ready pose and overshoots. Lowering
                // it: square back up.
                target = IsShouldered(next) ? _stanceYaw : 0f;
            }

            target = Mathf.Clamp(target, -MaxStanceDegrees, MaxStanceDegrees);
            _stanceYaw = Mathf.Lerp(_stanceYaw, target, 1f - Mathf.Exp(-StanceTurnRate * Time.deltaTime));
            _root.localRotation = Quaternion.Euler(0f, _stanceYaw, 0f);
        }

        private static bool IsShouldered(int stateHash) => stateHash == AimStateId || stateHash == FireStateId;

        private bool RifleStateIs(int stateHash)
        {
            if (RifleLayerIndex <= 0 || RifleLayerIndex >= _animator.layerCount) return false;

            if (_animator.GetCurrentAnimatorStateInfo(RifleLayerIndex).shortNameHash == stateHash) return true;
            return _animator.IsInTransition(RifleLayerIndex) &&
                   _animator.GetNextAnimatorStateInfo(RifleLayerIndex).shortNameHash == stateHash;
        }

        private void DetectReload()
        {
            if (_weapon == null) return;

            if (_weapon.IsReloading && !_wasReloading && _animator != null)
                _animator.SetTrigger(ReloadId);

            _wasReloading = _weapon.IsReloading;
        }

        private void DetectHit()
        {
            if (_health == null || _animator == null) return;
            if (_health.LastDamagedAt <= _lastHitAt) return;

            _lastHitAt = _health.LastDamagedAt;
            if (_health.IsAlive) _animator.SetTrigger(HitId);
        }

        private void DetectLocomotionChange()
        {
            var state = _motor.State;
            if (state == _lastLocomotion) return;

            if (_animator != null)
            {
                if (state == ReflexMotor.Locomotion.Vaulting) _animator.SetTrigger(VaultId);
                else if (state == ReflexMotor.Locomotion.Sliding) _animator.SetTrigger(SlideId);
            }

            _lastLocomotion = state;
        }

        /// <summary>
        /// Peeking is a spine rotation, not a clip. Written after the Animator has
        /// posed the skeleton so it layers over whatever is playing -- and because
        /// it is driven by the same intent the brain chose, it can never fall out
        /// of sync with where the agent is actually looking.
        /// </summary>
        private void ApplyLean(bool aiming)
        {
            if (_spine == null) return;

            // A body now stays on screen after death, and its last intent with it;
            // a corpse must not keep peeking.
            float target = 0f;
            if (!_agent.IsAlive) target = 0f;
            else if (_agent.CurrentIntent == Core.TacticalIntent.Peek) target = 1f;
            else if (aiming && _motor.IsCrouched) target = 0.35f;

            _lean = Mathf.Lerp(_lean, target, Time.deltaTime * 6f);
            if (Mathf.Abs(_lean) < 0.01f) return;

            _spine.localRotation *= Quaternion.Euler(0f, 0f, _lean * 18f);
        }

        private void AnimateMascot(float speed01, bool aiming)
        {
            if (_root == null) return;

            if (!_agent.IsAlive)
            {
                PlayDeath();
                return;
            }

            float dt = Time.deltaTime;

            // Step phase advances with distance covered, not with time, so the legs
            // stay in sync with the ground at any speed.
            _stepPhase += speed01 * _motor.MoveSpeed * StepsPerMetre * Mathf.PI * 2f * dt;

            float swing = Mathf.Sin(_stepPhase) * 38f * speed01;
            _legLeft.localRotation = Quaternion.Euler(swing, 0f, 0f);
            _legRight.localRotation = Quaternion.Euler(-swing, 0f, 0f);

            float crouch = _motor.IsCrouched ? 1f : 0f;
            float bob = Mathf.Abs(Mathf.Cos(_stepPhase)) * 0.045f * speed01;

            _root.localPosition = Vector3.Lerp(_root.localPosition,
                new Vector3(0f, bob - crouch * 0.3f, 0f), dt * 9f);

            _root.localScale = Vector3.Lerp(_root.localScale,
                new Vector3(1f, 1f - crouch * 0.18f, 1f), dt * 9f);

            // Lean into the direction of travel; it reads as intent at a distance.
            float lean = LeanDegrees * speed01;
            _root.localRotation = Quaternion.Lerp(_root.localRotation,
                Quaternion.Euler(lean, 0f, 0f), dt * 8f);

            AnimateUpperBody(speed01, aiming, dt);
        }

        private void AnimateUpperBody(float speed01, bool aiming, float dt)
        {
            // Arms counter-swing while moving, then snap forward onto the rifle the
            // moment we have something to shoot at.
            float armSwing = Mathf.Sin(_stepPhase) * 26f * speed01 * (aiming ? 0.2f : 1f);

            var leftTarget = aiming
                ? Quaternion.Euler(-72f, 0f, 20f)
                : Quaternion.Euler(-armSwing, 0f, 16f);
            var rightTarget = aiming
                ? Quaternion.Euler(-78f, 0f, -12f)
                : Quaternion.Euler(armSwing, 0f, -16f);

            _armLeft.localRotation = Quaternion.Lerp(_armLeft.localRotation, leftTarget, dt * 10f);
            _armRight.localRotation = Quaternion.Lerp(_armRight.localRotation, rightTarget, dt * 10f);

            var gunTarget = aiming
                ? new Vector3(0.14f, 1.32f, 0.58f)
                : new Vector3(0.24f, 1.1f, 0.3f);

            _gun.localPosition = Vector3.Lerp(_gun.localPosition,
                gunTarget - Vector3.forward * (_recoil * 0.16f), dt * 12f);

            _gun.localRotation = Quaternion.Lerp(_gun.localRotation,
                Quaternion.Euler(aiming ? -_recoil * 14f : 26f, 0f, 0f), dt * 12f);

            // Head keeps looking level while the body leans.
            _head.localRotation = Quaternion.Lerp(_head.localRotation,
                Quaternion.Euler(-LeanDegrees * speed01, 0f, 0f), dt * 8f);

            _torso.localRotation = Quaternion.Lerp(_torso.localRotation,
                Quaternion.Euler(aiming ? 6f : 0f, 0f, 0f), dt * 8f);
        }

        private void PlayDeath()
        {
            _root.localRotation = Quaternion.Lerp(_root.localRotation,
                Quaternion.Euler(84f, 0f, 0f), Time.deltaTime * 6f);

            _root.localPosition = Vector3.Lerp(_root.localPosition,
                new Vector3(0f, -0.35f, 0f), Time.deltaTime * 6f);
        }

        /// <summary>Put the body back upright for the next round.</summary>
        public void ResetVisual()
        {
            _deathPlayed = false;
            _recoil = 0f;
            _lastAmmo = int.MinValue;
            _wasReloading = false;
            _lastHitAt = float.NegativeInfinity;
            _lastLocomotion = ReflexMotor.Locomotion.Normal;
            _lean = 0f;
            _rifleWeight = 1f;
            _gripWeight = 1f;
            _stanceYaw = 0f;

            // Rebind drops the skeleton into its bind pose, and it would render that
            // way for a frame: a T-pose with the rifle standing straight up out of
            // the hand. Bodies now stay on screen through the respawn, so evaluate
            // the default pose right away.
            if (_animator != null)
            {
                _animator.Rebind();
                _animator.Update(0f);
            }

            if (_root == null) return;
            _root.localPosition = Vector3.zero;
            _root.localRotation = Quaternion.identity;
            _root.localScale = Vector3.one;
        }
    }
}
