using System;
using UnityEngine;

namespace JevNpcBrain.Combat
{
    /// <summary>
    /// Health, and the record of who last hurt us and when.
    ///
    /// Identical settings for both teams -- the fairness contract means a round is
    /// decided by decisions, never by stats.
    /// </summary>
    public sealed class Damageable : MonoBehaviour
    {
        public float MaxHealth = 100f;

        public float Health { get; private set; }
        public bool IsAlive => Health > 0f;
        public float Health01 => Mathf.Clamp01(Health / Mathf.Max(MaxHealth, 1f));

        public float LastDamagedAt { get; private set; } = float.NegativeInfinity;
        public Vector3 LastDamageDirection { get; private set; }

        /// <summary>Raised with the killer, if there was one.</summary>
        public event Action<GameObject> Died;

        /// <summary>Raised on every hit with (attacker, damage).</summary>
        public event Action<GameObject, float> Damaged;

        private void Awake() => Health = MaxHealth;

        public void ResetHealth()
        {
            Health = MaxHealth;
            LastDamagedAt = float.NegativeInfinity;
            LastDamageDirection = Vector3.zero;
        }

        public void TakeDamage(float amount, GameObject attacker, Vector3 fromDirection)
        {
            if (!IsAlive || amount <= 0f) return;

            Health = Mathf.Max(0f, Health - amount);
            LastDamagedAt = Time.time;
            LastDamageDirection = fromDirection;

            Damaged?.Invoke(attacker, amount);
            if (!IsAlive) Died?.Invoke(attacker);
        }
    }
}
