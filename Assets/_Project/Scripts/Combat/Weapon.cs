using UnityEngine;

namespace JevNpcBrain.Combat
{
    using Core;
    using Perception;

    /// <summary>
    /// Hitscan rifle with a magazine, a reload and a spread cone.
    ///
    /// No projectile travel time, on purpose. We want prediction to be rewarded in
    /// the tactical layer -- reading where someone will be, when to push, when a
    /// reload has opened a window -- not in the aiming layer, where it would just
    /// measure who leads their shots better and drown the signal we care about.
    /// </summary>
    public sealed class Weapon : MonoBehaviour
    {
        [Header("Ballistics")]
        public float Damage = 18f;
        public float Range = 45f;
        public float RoundsPerSecond = 5f;

        [Tooltip("Half-angle of the spread cone, degrees. Grows while moving.")]
        public float BaseSpread = 1.2f;
        public float MovingSpreadPenalty = 2.5f;

        [Header("Magazine")]
        public int MagazineSize = 24;
        public float ReloadSeconds = 2.0f;

        [Header("Audibility")]
        public float ShotLoudness = 1f;

        [Header("Suppression")]
        [Tooltip("A round passing this close to an enemy without hitting them pins them down.")]
        public float NearMissRadius = 1.5f;

        [Tooltip("Suppression added by a round that grazes the target; less the wider it passes.")]
        public float NearMissSuppression = 0.35f;

        public int Ammo { get; private set; }
        public bool IsReloading { get; private set; }
        public float Ammo01 => Mathf.Clamp01(Ammo / (float)Mathf.Max(MagazineSize, 1));
        public float ReloadProgress01 => IsReloading
            ? Mathf.Clamp01(1f - (_reloadDoneAt - Time.time) / Mathf.Max(ReloadSeconds, 0.01f))
            : 0f;

        /// <summary>Set by the owning agent so gunfire is attributed to a team.</summary>
        public int Team;

        public LayerMask HitMask = ~0;

        private float _nextShotAt;
        private float _reloadDoneAt;

        private void Awake() => Ammo = MagazineSize;

        public void ResetWeapon()
        {
            Ammo = MagazineSize;
            IsReloading = false;
            _nextShotAt = 0f;
        }

        private void Update()
        {
            if (IsReloading && Time.time >= _reloadDoneAt)
            {
                IsReloading = false;
                Ammo = MagazineSize;
            }
        }

        public void BeginReload()
        {
            if (IsReloading || Ammo >= MagazineSize) return;
            IsReloading = true;
            _reloadDoneAt = Time.time + ReloadSeconds;
        }

        public bool CanFire => !IsReloading && Ammo > 0 && Time.time >= _nextShotAt;

        /// <summary>
        /// Fires one round from origin toward target. Returns true if a round left
        /// the barrel, whether or not it connected.
        /// </summary>
        public bool TryFire(Vector3 origin, Vector3 target, float moveSpeed01, GameObject owner)
        {
            if (!CanFire)
            {
                if (Ammo <= 0) BeginReload();
                return false;
            }

            _nextShotAt = Time.time + 1f / Mathf.Max(RoundsPerSecond, 0.1f);
            Ammo--;

            var direction = (target - origin).normalized;
            float spread = BaseSpread + MovingSpreadPenalty * Mathf.Clamp01(moveSpeed01);
            direction = Quaternion.Euler(
                Random.Range(-spread, spread),
                Random.Range(-spread, spread),
                0f) * direction;

            NoiseEvents.Report(origin, ShotLoudness, Team, Time.time);

            float travelled = Range;
            Damageable struck = null;

            if (Physics.Raycast(origin, direction, out var hit, Range, HitMask, QueryTriggerInteraction.Ignore))
            {
                travelled = hit.distance;
                var victim = hit.collider.GetComponentInParent<Damageable>();
                if (victim != null && victim.gameObject != owner)
                {
                    struck = victim;
                    victim.TakeDamage(Damage, owner, -direction);
                }
            }

            SuppressNearMisses(origin, direction, travelled, struck);

            if (Ammo <= 0) BeginReload();
            return true;
        }

        /// <summary>
        /// The only writer of NpcAgent suppression -- the self vector declared a
        /// Suppression01 field from the start, but nothing ever filled it, so every
        /// observation carried a constant zero there. The baseline never reads it,
        /// so this changes what the learned brain can see, not how the baseline plays.
        /// </summary>
        private void SuppressNearMisses(Vector3 origin, Vector3 direction, float length, Damageable struck)
        {
            var agents = NpcAgent.All;

            for (int i = 0; i < agents.Count; i++)
            {
                var agent = agents[i];
                if (agent.Team == Team || !agent.IsAlive || agent.Health == struck) continue;

                var toChest = agent.transform.position + Vector3.up * 1.2f - origin;
                float along = Vector3.Dot(toChest, direction);
                if (along <= 0f || along > length) continue;

                float miss = (toChest - direction * along).magnitude;
                if (miss >= NearMissRadius) continue;

                agent.AddSuppression(NearMissSuppression * (1f - miss / NearMissRadius));
            }
        }
    }
}
