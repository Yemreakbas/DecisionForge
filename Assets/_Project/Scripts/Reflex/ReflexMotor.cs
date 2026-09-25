using UnityEngine;

namespace JevNpcBrain.Reflex
{
    using Arena;
    using Combat;
    using Core;
    using Perception;

    /// <summary>
    /// The 60 Hz layer. Turns one <see cref="TacticalIntent"/> into footsteps,
    /// a facing and a trigger pull.
    ///
    /// Everything here is hand-written and deterministic, and it is identical for
    /// both teams. That is deliberate: if the learned brain were allowed to move
    /// or aim differently, a win could come from twitch rather than from judgement,
    /// and the experiment would be measuring the wrong thing.
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    public sealed class ReflexMotor : MonoBehaviour
    {
        public enum Locomotion { Normal, Vaulting, Sliding }

        [Header("Movement")]
        public float MoveSpeed = 4.6f;
        public float SprintSpeed = 6.8f;
        public float CrouchSpeed = 2.3f;
        public float Acceleration = 22f;
        public float TurnDegreesPerSecond = 540f;
        public float ArriveRadius = 1.1f;

        [Header("Aim")]
        public Transform Eye;
        public float AimToleranceDegrees = 9f;

        [Header("Sprint")]
        [Tooltip("Sprint only when disengaged and the destination is far enough to be worth it.")]
        public float SprintMinDistance = 7f;

        [Header("Vault")]
        [Tooltip("Off until the animation set covers it. The behaviour works and is " +
                 "tested; without a clip the agent would slide over cover unanimated. " +
                 "Leaving it off also keeps the 19-21 calibration valid, since that " +
                 "was measured before vaulting existed.")]
        public bool EnableVault = false;

        [Tooltip("Crouch-height cover within this range gets vaulted instead of walked around.")]
        public float VaultProbeDistance = 1.9f;
        public float VaultSeconds = 0.55f;
        public float VaultHeight = 1.25f;

        [Header("Slide")]
        [Tooltip("Off until the animation set covers it. See EnableVault.")]
        public bool EnableSlide = false;

        [Tooltip("Slide into cover when arriving this fast, from this far out.")]
        public float SlideTriggerDistance = 3.2f;
        public float SlideMinSpeed01 = 0.8f;
        public float SlideSeconds = 0.6f;

        [Header("Stuck recovery")]
        public float StuckSpeedThreshold = 0.4f;
        public float StuckSeconds = 1.0f;

        private CharacterController _controller;
        private Weapon _weapon;

        private Vector3 _destination;
        private Vector3 _velocity;
        private Vector3 _aimPoint;
        private bool _hasAimPoint;
        private bool _wantsFire;
        private bool _crouched;

        private float _stuckTimer;
        private Vector3 _sidestep;

        private Locomotion _locomotion = Locomotion.Normal;
        private float _stateTimer;
        private Vector3 _stateFrom;
        private Vector3 _stateTo;
        private float _slideCooldown;

        public TacticalIntent CurrentIntent { get; private set; }
        public Vector3 Destination => _destination;
        public bool IsCrouched => _crouched;
        public float Speed01 => Mathf.Clamp01(_velocity.magnitude / Mathf.Max(MoveSpeed, 0.1f));

        /// <summary>
        /// Planar velocity in the agent's own frame, x = strafe, y = forward, each
        /// normalised to sprint speed. This is what a 2D locomotion blend tree
        /// needs: without it an agent that is aiming at an enemy while moving
        /// sideways plays a forward run and looks like it is skating.
        /// </summary>
        public Vector2 MoveLocal { get; private set; }

        public bool IsSprinting { get; private set; }
        public bool IsVaulting => _locomotion == Locomotion.Vaulting;
        public bool IsSliding => _locomotion == Locomotion.Sliding;
        public bool IsGrounded => _controller != null && _controller.isGrounded;
        public Locomotion State => _locomotion;

        /// <summary>True when we have been trying to move and failing.</summary>
        public bool IsStuck => _stuckTimer >= StuckSeconds;

        private void Awake()
        {
            _controller = GetComponent<CharacterController>();
            _weapon = GetComponentInChildren<Weapon>();
            _destination = transform.position;
            if (Eye == null) Eye = transform;
        }

        public void ResetMotor(Vector3 position, Quaternion rotation)
        {
            _controller.enabled = false;
            transform.SetPositionAndRotation(position, rotation);
            _controller.enabled = true;

            _destination = position;
            _velocity = Vector3.zero;
            _stuckTimer = 0f;
            _hasAimPoint = false;
            _wantsFire = false;
            _crouched = false;

            _locomotion = Locomotion.Normal;
            _stateTimer = 0f;
            _slideCooldown = 0f;
            IsSprinting = false;
            MoveLocal = Vector2.zero;
        }

        /// <summary>
        /// Called once per tactical tick. Translates the intent into a destination
        /// and a stance; the per-frame work happens in Update.
        /// </summary>
        public void ApplyIntent(TacticalIntent intent, TacticalObservation obs, Vector3 allyPosition)
        {
            CurrentIntent = intent;

            var layout = ArenaLayout.Current;
            var self = transform.position;
            var s = obs.Summary;

            var threatAxis = s.EnemyBeliefDistance < 900f
                ? Flat(s.EnemyBeliefCentroid - self).normalized
                : transform.forward;

            switch (intent)
            {
                case TacticalIntent.Hold:
                case TacticalIntent.Suppress:
                    _destination = self;
                    _crouched = s.CoverQualityHere > 0.35f;
                    break;

                case TacticalIntent.PushCoverForward:
                    _destination = s.HasBestCover ? s.BestCoverPosition : self + threatAxis * 4f;
                    _crouched = false;
                    break;

                case TacticalIntent.PushCoverFlank:
                    _destination = s.HasFlankCover ? s.FlankCoverPosition : self + threatAxis * 3f;
                    _crouched = false;
                    break;

                case TacticalIntent.Retreat:
                    _destination = self - threatAxis * 7f;
                    _crouched = false;
                    break;

                case TacticalIntent.FlankLeft:
                    _destination = self + Perpendicular(threatAxis) * 8f + threatAxis * 2f;
                    _crouched = false;
                    break;

                case TacticalIntent.FlankRight:
                    _destination = self - Perpendicular(threatAxis) * 8f + threatAxis * 2f;
                    _crouched = false;
                    break;

                case TacticalIntent.Peek:
                    // Lean out just far enough to see, then the next tick decides.
                    _destination = self + Perpendicular(threatAxis) * 1.8f;
                    _crouched = false;
                    break;

                case TacticalIntent.RegroupAlly:
                    _destination = s.HasAlly ? allyPosition : self;
                    _crouched = false;
                    break;

                case TacticalIntent.ContestCenter:
                    _destination = layout != null ? layout.CenterPosition : self;
                    _crouched = false;
                    break;
            }

            _destination = ClampToArena(_destination, layout);
        }

        /// <summary>Set by the agent each tick: the enemy we can actually see, if any.</summary>
        public void SetFiringSolution(Vector3 aimPoint, bool hasTarget)
        {
            _aimPoint = aimPoint;
            _hasAimPoint = hasTarget;
            _wantsFire = hasTarget;
        }

        /// <summary>Suppression keeps the muzzle pointed at a believed position.</summary>
        public void SetSuppressionTarget(Vector3 aimPoint)
        {
            _aimPoint = aimPoint;
            _hasAimPoint = true;
            _wantsFire = true;
        }

        private void Update()
        {
            float dt = Time.deltaTime;
            _slideCooldown = Mathf.Max(0f, _slideCooldown - dt);

            switch (_locomotion)
            {
                case Locomotion.Vaulting: TickVault(dt); break;
                case Locomotion.Sliding: TickSlide(dt); break;
                default: Steer(dt); break;
            }

            PublishLocalMotion();
            Face(dt);

            // Hands are busy going over a wall; holding the trigger through a vault
            // would both look wrong and hand the mover free shots.
            if (_locomotion != Locomotion.Vaulting) Shoot();
        }

        /// <summary>
        /// Projects world velocity into the agent frame for the blend tree, and
        /// scales by sprint speed so the full range is reachable.
        /// </summary>
        private void PublishLocalMotion()
        {
            var local = transform.InverseTransformDirection(_velocity);
            float scale = Mathf.Max(SprintSpeed, 0.1f);
            MoveLocal = Vector2.ClampMagnitude(new Vector2(local.x, local.z) / scale, 1f);
        }

        private void Steer(float dt)
        {
            var toTarget = Flat(_destination - transform.position);
            float distance = toTarget.magnitude;

            Vector3 desired;
            if (distance <= ArriveRadius)
            {
                desired = Vector3.zero;
                _stuckTimer = 0f;
                IsSprinting = false;
            }
            else
            {
                var heading = toTarget / distance;

                if (TryStartVault(heading)) return;
                if (TryStartSlide(heading, distance)) return;

                var direction = Avoid(heading);

                // Sprinting is a disengaged move: head down, gun lowered. Doing it
                // while aiming would let an agent close distance at full speed with
                // a firing solution, which no shooter would allow.
                IsSprinting = !_crouched && !_hasAimPoint && distance > SprintMinDistance;

                float speed = _crouched ? CrouchSpeed : (IsSprinting ? SprintSpeed : MoveSpeed);
                desired = direction * speed;
            }

            _velocity = Vector3.MoveTowards(_velocity, desired, Acceleration * dt);

            // Gravity is only here to keep the controller grounded; the arena is flat.
            var motion = _velocity * dt + Vector3.down * 9.81f * dt * dt;
            _controller.Move(motion);

            if (distance > ArriveRadius && _velocity.magnitude < StuckSpeedThreshold)
            {
                _stuckTimer += dt;
                if (_stuckTimer >= StuckSeconds && _sidestep == Vector3.zero)
                    _sidestep = Perpendicular(Flat(_destination - transform.position).normalized)
                                * (Random.value < 0.5f ? 1f : -1f);
            }
            else
            {
                _stuckTimer = 0f;
                _sidestep = Vector3.zero;
            }
        }

        /// <summary>
        /// Three-probe obstacle avoidance against the static occupancy grid. Cheap,
        /// and good enough for a convex greybox arena -- an agent that clips a
        /// corner here is a dumb moment the metrics will catch, not a crash.
        /// </summary>
        private Vector3 Avoid(Vector3 direction)
        {
            if (_sidestep != Vector3.zero) return (direction + _sidestep * 1.5f).normalized;

            var layout = ArenaLayout.Current;
            if (layout == null) return direction;

            const float probe = 2.2f;
            var origin = transform.position;

            if (layout.SampleHeight(origin + direction * probe) < 0.5f) return direction;

            var left = Quaternion.Euler(0f, -45f, 0f) * direction;
            var right = Quaternion.Euler(0f, 45f, 0f) * direction;

            bool leftClear = layout.SampleHeight(origin + left * probe) < 0.5f;
            bool rightClear = layout.SampleHeight(origin + right * probe) < 0.5f;

            if (leftClear && !rightClear) return left;
            if (rightClear && !leftClear) return right;
            if (leftClear) return Random.value < 0.5f ? left : right;

            return Quaternion.Euler(0f, 90f, 0f) * direction;
        }

        private void Face(float dt)
        {
            Vector3 lookDirection;

            if (_hasAimPoint) lookDirection = Flat(_aimPoint - transform.position);
            else if (_velocity.sqrMagnitude > 0.05f) lookDirection = Flat(_velocity);
            else return;

            if (lookDirection.sqrMagnitude < 0.001f) return;

            transform.rotation = Quaternion.RotateTowards(
                transform.rotation,
                Quaternion.LookRotation(lookDirection),
                TurnDegreesPerSecond * dt);
        }

        private void Shoot()
        {
            if (!_wantsFire || _weapon == null || !_hasAimPoint) return;

            var toTarget = Flat(_aimPoint - transform.position);
            if (toTarget.sqrMagnitude < 0.001f) return;

            // Do not fire until the barrel is roughly on target, otherwise an agent
            // spraying mid-turn would look broken and skew the accuracy numbers.
            if (Vector3.Angle(Flat(transform.forward), toTarget) > AimToleranceDegrees) return;

            _weapon.TryFire(Eye.position, _aimPoint, Speed01, gameObject);
        }

        /// <summary>
        /// Go over crouch-height cover instead of around it.
        ///
        /// The occupancy grid already encodes half cover as 0.5, so spotting a
        /// vaultable obstacle costs two samples and no raycasts. We only commit if
        /// the far side is clear -- vaulting into a wall would be worse than the
        /// detour we are avoiding.
        /// </summary>
        private bool TryStartVault(Vector3 heading)
        {
            if (!EnableVault) return false;
            if (_crouched || _locomotion != Locomotion.Normal) return false;
            if (_velocity.magnitude < MoveSpeed * 0.45f) return false;

            var layout = ArenaLayout.Current;
            if (layout == null) return false;

            var origin = transform.position;
            float ahead = layout.SampleHeight(origin + heading * VaultProbeDistance);
            if (ahead < 0.35f || ahead > 0.9f) return false;

            var landing = origin + heading * (VaultProbeDistance + 1.6f);
            if (layout.SampleHeight(landing) > 0.2f) return false;

            _locomotion = Locomotion.Vaulting;
            _stateTimer = 0f;
            _stateFrom = origin;
            _stateTo = landing;
            _hasAimPoint = false;
            _wantsFire = false;

            // Moved by transform for the duration, so the capsule must not fight it.
            _controller.enabled = false;
            return true;
        }

        private void TickVault(float dt)
        {
            _stateTimer += dt;
            float t = Mathf.Clamp01(_stateTimer / Mathf.Max(VaultSeconds, 0.05f));

            var flat = Vector3.Lerp(_stateFrom, _stateTo, t);
            flat.y = Mathf.Lerp(_stateFrom.y, _stateTo.y, t) + Mathf.Sin(t * Mathf.PI) * VaultHeight;
            transform.position = flat;

            if (t < 1f) return;

            _controller.enabled = true;
            _locomotion = Locomotion.Normal;

            var heading = Flat(_stateTo - _stateFrom).normalized;
            _velocity = heading * MoveSpeed * 0.6f;
            _stuckTimer = 0f;
        }

        /// <summary>
        /// Slide the last couple of metres into cover rather than jogging to a stop.
        /// Purely a reflex flourish -- it does not change where the agent ends up,
        /// only how fast it gets low, which is exactly the kind of thing that reads
        /// as competence to a viewer.
        /// </summary>
        private bool TryStartSlide(Vector3 heading, float distance)
        {
            if (!EnableSlide) return false;
            if (_locomotion != Locomotion.Normal) return false;
            if (_slideCooldown > 0f || _crouched) return false;
            if (Speed01 < SlideMinSpeed01) return false;
            if (distance > SlideTriggerDistance || distance < ArriveRadius) return false;

            _locomotion = Locomotion.Sliding;
            _stateTimer = 0f;
            _stateFrom = transform.position;
            _stateTo = _destination;
            _crouched = true;
            return true;
        }

        private void TickSlide(float dt)
        {
            _stateTimer += dt;
            float t = Mathf.Clamp01(_stateTimer / Mathf.Max(SlideSeconds, 0.05f));

            // Decelerate hard, the way a real slide scrubs speed.
            float speed = Mathf.Lerp(SprintSpeed, 0.4f, t * t);
            var heading = Flat(_stateTo - transform.position);

            if (heading.sqrMagnitude > 0.01f)
            {
                _velocity = heading.normalized * speed;
                _controller.Move(_velocity * dt + Vector3.down * 9.81f * dt * dt);
            }

            if (t < 1f) return;

            _locomotion = Locomotion.Normal;
            _slideCooldown = 2.5f;
            _velocity = Vector3.zero;
        }

        private static Vector3 Flat(Vector3 v) => new Vector3(v.x, 0f, v.z);

        private static Vector3 Perpendicular(Vector3 v) => new Vector3(-v.z, 0f, v.x);

        private static Vector3 ClampToArena(Vector3 point, ArenaLayout layout)
        {
            if (layout == null) return point;

            var local = point - layout.Origin;
            local.y = 0f;

            float limit = layout.Radius - 2f;
            if (local.magnitude > limit) local = local.normalized * limit;

            return layout.Origin + local;
        }
    }
}
