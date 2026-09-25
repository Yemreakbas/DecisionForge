using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Perception
{
    /// <summary>
    /// What one agent thinks it knows about the enemy.
    ///
    /// This type is the single reason the comparison is meaningful. Neither brain
    /// ever reads an enemy transform; both read this. When contact is lost the
    /// belief dead-reckons forward, loses confidence and spreads out -- so
    /// "where will they be in a second" becomes a real question with a real cost
    /// for getting it wrong, which is exactly the question a learned predictor is
    /// supposed to answer better than a hand-written rule.
    /// </summary>
    public sealed class EnemyBelief
    {
        public struct Track
        {
            public int Id;
            public Vector3 Position;
            public Vector3 Velocity;
            public float Confidence;
            public float Spread;
            public float LastSeen;
            public bool VisibleNow;
        }

        /// <summary>Seconds for confidence to halve once contact is lost.</summary>
        public float HalfLife = 3.5f;

        /// <summary>Metres per second the belief blurs while unobserved.</summary>
        public float SpreadRate = 1.6f;

        public float MaxSpread = 9f;
        public float ForgetBelow = 0.04f;

        private readonly List<Track> _tracks = new List<Track>(8);
        private readonly List<NoiseEvents.Noise> _noises = new List<NoiseEvents.Noise>(16);

        public IReadOnlyList<Track> Tracks => _tracks;

        public void Clear() => _tracks.Clear();

        /// <summary>A direct sighting. Resets confidence and collapses the spread.</summary>
        public void Observe(int id, Vector3 position, float time)
        {
            for (int i = 0; i < _tracks.Count; i++)
            {
                if (_tracks[i].Id != id) continue;

                var t = _tracks[i];
                float dt = Mathf.Max(time - t.LastSeen, 0.0001f);
                t.Velocity = Vector3.Lerp(t.Velocity, (position - t.Position) / dt, 0.5f);
                t.Position = position;
                t.Confidence = 1f;
                t.Spread = 0.5f;
                t.LastSeen = time;
                t.VisibleNow = true;
                _tracks[i] = t;
                return;
            }

            _tracks.Add(new Track
            {
                Id = id,
                Position = position,
                Velocity = Vector3.zero,
                Confidence = 1f,
                Spread = 0.5f,
                LastSeen = time,
                VisibleNow = true
            });
        }

        /// <summary>
        /// Ages every track that was not seen this tick. Called once per tactical
        /// tick, after all Observe calls.
        /// </summary>
        public void Age(float time, float deltaTime, int listenerTeam)
        {
            float decay = Mathf.Pow(0.5f, deltaTime / Mathf.Max(HalfLife, 0.01f));

            for (int i = _tracks.Count - 1; i >= 0; i--)
            {
                var t = _tracks[i];

                if (!t.VisibleNow)
                {
                    // Dead reckoning: keep moving them the way they were going,
                    // but admit we are less sure with every metre.
                    t.Position += t.Velocity * deltaTime;
                    t.Confidence *= decay;
                    t.Spread = Mathf.Min(t.Spread + SpreadRate * deltaTime, MaxSpread);
                }

                t.VisibleNow = false;
                _tracks[i] = t;

                if (t.Confidence < ForgetBelow) _tracks.RemoveAt(i);
            }

            FoldInNoise(time, listenerTeam);
        }

        /// <summary>
        /// A gunshot we could not see is still evidence. It refreshes the nearest
        /// stale track, or creates a low-confidence one if we had nothing.
        /// </summary>
        private void FoldInNoise(float time, int listenerTeam)
        {
            NoiseEvents.Collect(listenerTeam, time, _noises);

            for (int n = 0; n < _noises.Count; n++)
            {
                var noise = _noises[n];
                float freshness = 1f - Mathf.Clamp01((time - noise.Time) / NoiseEvents.Lifetime);
                float weight = Mathf.Clamp01(noise.Loudness * freshness);
                if (weight < 0.05f) continue;

                int best = -1;
                float bestDistance = 12f;

                for (int i = 0; i < _tracks.Count; i++)
                {
                    float d = Vector3.Distance(_tracks[i].Position, noise.Position);
                    if (d >= bestDistance) continue;
                    bestDistance = d;
                    best = i;
                }

                if (best >= 0)
                {
                    var t = _tracks[best];
                    if (t.Confidence < weight)
                    {
                        t.Position = Vector3.Lerp(t.Position, noise.Position, 0.6f);
                        t.Confidence = weight;
                        t.Spread = Mathf.Min(t.Spread, 3.5f);
                        _tracks[best] = t;
                    }
                }
                else
                {
                    _tracks.Add(new Track
                    {
                        Id = -1 - n,
                        Position = noise.Position,
                        Velocity = Vector3.zero,
                        Confidence = weight * 0.7f,
                        Spread = 3.5f,
                        LastSeen = noise.Time
                    });
                }
            }
        }

        public float TotalMass()
        {
            float sum = 0f;
            for (int i = 0; i < _tracks.Count; i++) sum += _tracks[i].Confidence;
            return sum;
        }

        public bool TryGetStrongest(out Track track)
        {
            track = default;
            float best = 0f;

            for (int i = 0; i < _tracks.Count; i++)
            {
                if (_tracks[i].Confidence <= best) continue;
                best = _tracks[i].Confidence;
                track = _tracks[i];
            }

            return best > 0f;
        }
    }
}
