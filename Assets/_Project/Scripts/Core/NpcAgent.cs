using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;

namespace JevNpcBrain.Core
{
    using Arena;
    using Combat;
    using Perception;
    using Reflex;
    using Tactical;

    /// <summary>
    /// One fighter. Owns the sensor, the belief, the brain and the reflex layer,
    /// and runs the tactical tick that connects them.
    ///
    /// The brain is injected rather than chosen here, because which brain an agent
    /// carries is the independent variable of the whole experiment. Everything
    /// else on this component is identical for both teams.
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    public sealed class NpcAgent : MonoBehaviour
    {
        public static readonly List<NpcAgent> All = new List<NpcAgent>();

        /// <summary>
        /// Raised after every tactical decision, once this tick's observation and
        /// intent are final. The phase 2 dataset recorder listens; nothing in the
        /// game does.
        /// </summary>
        public static event System.Action<NpcAgent> Decided;

        [Header("Identity")]
        public int Team;
        public bool IsCaptain;

        /// <summary>
        /// Stable small index assigned by MatchDirector. Deliberately not the
        /// Unity instance id: phase 2 logs observations to disk, and those logs are
        /// only comparable across runs if an agent keeps the same id every time.
        /// </summary>
        public int AgentId;
        public string DisplayName = "Agent";

        [Header("Tick")]
        [Tooltip("Tactical decisions per second. Same for both brains, by contract.")]
        public float TacticalHz = 10f;

        [Header("Wiring")]
        public Damageable Health;
        public Weapon Weapon;
        public ReflexMotor Motor;
        public Transform Eye;

        /// <summary>Written by MatchDirector so the self vector can see the scoreline.</summary>
        [HideInInspector] public float TeamScore01;
        [HideInInspector] public float EnemyScore01;
        [HideInInspector] public float MatchTimeRemaining01 = 1f;

        public ITacticalBrain Brain { get; private set; }
        public TacticalObservation Observation { get; } = new TacticalObservation();
        public EnemyBelief Belief { get; } = new EnemyBelief();
        public TacticalIntent CurrentIntent { get; private set; }

        /// <summary>Wall-clock cost of the last Decide call. Published, not hidden.</summary>
        public float LastDecisionMilliseconds { get; private set; }

        public bool IsAlive => Health != null && Health.IsAlive;
        public Vector3 EyePosition => Eye != null ? Eye.position : transform.position + Vector3.up * 1.5f;

        private readonly SensorGrid _sensor = new SensorGrid();
        private readonly List<Vector3> _allyPositions = new List<Vector3>(4);
        private readonly Stopwatch _stopwatch = new Stopwatch();

        private float _tickTimer;
        private float _tickInterval;
        private float _lastEnemySeenAt = float.NegativeInfinity;
        private float _suppression;
        private float _intentChangedAt;

        private void Awake()
        {
            if (Health == null) Health = GetComponent<Damageable>();
            if (Weapon == null) Weapon = GetComponentInChildren<Weapon>();
            if (Motor == null) Motor = GetComponent<ReflexMotor>();
            if (Eye == null) Eye = transform;

            _tickInterval = 1f / Mathf.Max(TacticalHz, 1f);

            // Stagger the ticks so six agents never all think on the same frame.
            _tickTimer = Random.value * _tickInterval;
        }

        private void OnEnable() => All.Add(this);
        private void OnDisable() => All.Remove(this);

        public void Configure(int agentId, int team, bool captain, string displayName, ITacticalBrain brain)
        {
            AgentId = agentId;
            Team = team;
            IsCaptain = captain;
            DisplayName = displayName;
            Brain = brain;

            if (Weapon != null) Weapon.Team = team;
        }

        public void ResetForRound(Vector3 position, Quaternion rotation)
        {
            Health.ResetHealth();
            Weapon.ResetWeapon();
            Motor.enabled = true;
            Motor.ResetMotor(position, rotation);
            Belief.Clear();
            Brain?.Reset();

            _lastEnemySeenAt = float.NegativeInfinity;
            _suppression = 0f;
            _intentChangedAt = Time.time;
            CurrentIntent = TacticalIntent.Hold;
            gameObject.SetActive(true);
        }

        private void Update()
        {
            if (!IsAlive || Brain == null || ArenaLayout.Current == null) return;

            _suppression = Mathf.MoveTowards(_suppression, 0f, Time.deltaTime * 0.5f);

            _tickTimer -= Time.deltaTime;
            if (_tickTimer > 0f) return;

            _tickTimer += _tickInterval;
            TacticalTick(_tickInterval);
        }

        private void TacticalTick(float dt)
        {
            float now = Time.time;

            var visibleEnemy = Look(now);
            Belief.Age(now, dt, Team);

            CollectAllies();
            _sensor.Populate(Observation, transform.position, transform.forward,
                Belief, _allyPositions, Team, now);

            FillSelfState(now);
            Observation.FinalizeSelfVector();

            _stopwatch.Restart();
            var intent = Brain.Decide(Observation);
            _stopwatch.Stop();
            LastDecisionMilliseconds = (float)_stopwatch.Elapsed.TotalMilliseconds;

            if (intent != CurrentIntent) _intentChangedAt = now;
            CurrentIntent = intent;
            Decided?.Invoke(this);

            var allyPosition = _allyPositions.Count > 0 ? _allyPositions[0] : transform.position;
            Motor.ApplyIntent(intent, Observation, allyPosition);

            // The reflex layer shoots at what the eyes can see. Suppression is the
            // one case where we knowingly fire at a belief rather than a sighting.
            if (visibleEnemy != null)
            {
                Motor.SetFiringSolution(visibleEnemy.EyePosition, true);
            }
            else if (intent == TacticalIntent.Suppress && Belief.TryGetStrongest(out var track))
            {
                Motor.SetSuppressionTarget(track.Position + Vector3.up * 1.4f);
            }
            else
            {
                Motor.SetFiringSolution(Vector3.zero, false);
            }
        }

        /// <summary>
        /// Simulated eyes. This is the ONLY place an enemy transform is read, and
        /// what it produces is a belief update -- never a fact handed to the brain.
        /// </summary>
        private NpcAgent Look(float now)
        {
            var layout = ArenaLayout.Current;
            var eye = EyePosition;
            var forward = transform.forward;
            float cos = Mathf.Cos(_sensor.SightHalfAngle * Mathf.Deg2Rad);

            NpcAgent closest = null;
            float closestDistance = float.MaxValue;

            for (int i = 0; i < All.Count; i++)
            {
                var other = All[i];
                if (other.Team == Team || !other.IsAlive) continue;

                var delta = other.transform.position - transform.position;
                delta.y = 0f;
                float distance = delta.magnitude;

                if (distance > _sensor.SightRange) continue;
                if (distance > 0.5f && Vector3.Dot(delta / distance, forward) < cos) continue;
                if (!layout.HasLineOfSight(eye, other.EyePosition)) continue;

                Belief.Observe(other.AgentId, other.transform.position, now);
                _lastEnemySeenAt = now;

                if (distance >= closestDistance) continue;
                closestDistance = distance;
                closest = other;
            }

            return closest;
        }

        private void CollectAllies()
        {
            _allyPositions.Clear();

            for (int i = 0; i < All.Count; i++)
            {
                var other = All[i];
                if (other == this || other.Team != Team || !other.IsAlive) continue;
                _allyPositions.Add(other.transform.position);
            }
        }

        private void FillSelfState(float now)
        {
            var layout = ArenaLayout.Current;
            ref var self = ref Observation.Self;

            self = default;
            self.Health01 = Health.Health01;
            self.Ammo01 = Weapon.Ammo01;
            self.IsReloading = Weapon.IsReloading ? 1f : 0f;
            self.ReloadProgress01 = Weapon.ReloadProgress01;
            self.Stance = Motor.IsCrouched ? 1f : 0f;
            self.Speed01 = Motor.Speed01;
            self.SetFacing(transform.forward);

            self.TimeSinceEnemySeen01 = Mathf.Clamp01((now - _lastEnemySeenAt) / _sensor.MemoryHorizon);
            self.TimeSinceDamaged01 = Mathf.Clamp01((now - Health.LastDamagedAt) / _sensor.MemoryHorizon);
            self.HasLineOfSight = now - _lastEnemySeenAt < 0.25f ? 1f : 0f;

            float visible = 0f;
            for (int i = 0; i < Belief.Tracks.Count; i++)
                if (Belief.Tracks[i].Confidence > 0.9f) visible += 1f;
            self.VisibleEnemies01 = Mathf.Clamp01(visible / 3f);

            float allyHealth = 0f;
            int allyCount = 0;
            for (int i = 0; i < All.Count; i++)
            {
                var other = All[i];
                if (other == this || other.Team != Team || !other.IsAlive) continue;
                allyHealth += other.Health.Health01;
                allyCount++;
            }

            self.AlliesAlive01 = Mathf.Clamp01(allyCount / 2f);
            self.AllyAverageHealth01 = allyCount > 0 ? allyHealth / allyCount : 0f;

            self.TeamScore01 = TeamScore01;
            self.EnemyScore01 = EnemyScore01;
            self.MatchTimeRemaining01 = MatchTimeRemaining01;

            var summary = Observation.Summary;
            self.InCover01 = summary.CoverQualityHere;
            self.NearestKnownEnemyDistance01 = Mathf.Clamp01(summary.EnemyBeliefDistance / 40f);
            self.Suppression01 = Mathf.Clamp01(_suppression);

            self.LastIntent01 = (int)CurrentIntent / (float)(TacticalIntents.Count - 1);
            self.IntentHoldTime01 = Mathf.Clamp01((now - _intentChangedAt) / 5f);

            self.DistanceToCenter01 = layout != null
                ? Mathf.Clamp01(summary.DistanceToCenter / layout.Radius)
                : 0f;
            self.InCenterZone = layout != null && layout.IsInCenterZone(transform.position) ? 1f : 0f;
        }

        /// <summary>
        /// Out of the round. The body stays where it fell -- its death animation is
        /// the payoff of every engagement on camera -- but it stops being a
        /// physical object: no collider to soak up shots meant for someone behind
        /// it, no motor steering a corpse. Gameplay stays identical to removing the
        /// agent outright, which is what the calibration was measured under.
        /// ResetForRound brings both back.
        /// </summary>
        public void GoLimp()
        {
            if (Motor != null) Motor.enabled = false;

            var controller = GetComponent<CharacterController>();
            if (controller != null) controller.enabled = false;
        }

        /// <summary>Called by the weapon system when we are shot at but not hit.</summary>
        public void AddSuppression(float amount) => _suppression = Mathf.Clamp01(_suppression + amount);
    }
}
