using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;

    /// <summary>
    /// The control group: a hand-authored utility AI of the kind a shipped game
    /// would actually use.
    ///
    /// This is deliberately NOT a strawman. It scores every intent against the same
    /// observation the learned brain sees, it weighs cover quality and threat
    /// exposure properly, it exploits the enemy reload window, and it commits to a
    /// decision with hysteresis so it does not dither. If the JEV brain cannot beat
    /// this, the JEV brain has not earned anything -- and a rigged baseline would
    /// make the entire comparison worthless.
    ///
    /// Its one structural limitation is the point of the experiment: every
    /// behaviour here had to be anticipated and written down by a human, and these
    /// weights were tuned against one arena.
    /// </summary>
    public sealed class UtilityBrain : ITacticalBrain
    {
        public string Label => "Utility";

        /// <summary>
        /// Personality knobs. Captains get a distinct profile from their squad --
        /// that is what makes the 1v1 duel read as two characters rather than two
        /// copies of the same function.
        /// </summary>
        [System.Serializable]
        public sealed class Weights
        {
            public float AggressionBias = 0f;
            public float LowHealthThreshold = 0.35f;
            public float CoverSeekUrgency = 1.2f;
            public float ExposurePenalty = 1.6f;
            public float CenterValue = 0.8f;
            public float RegroupValue = 0.6f;
            public float LostContactCuriosity = 0.5f;
            public float CommitmentBonus = 0.35f;
            public float MinCommitSeconds = 0.6f;

            [Tooltip("Hold bonus while in contact with a loaded weapon. 0.5 is the calibrated baseline.")]
            public float ContactHoldBonus = 0.5f;

            public static Weights Captain() => new Weights
            {
                AggressionBias = 0.15f,
                CommitmentBonus = 0.45f,
                LostContactCuriosity = 0.7f
            };

            /// <summary>
            /// The control for JEV's first A/B win: the weapon spreads 3x wider on
            /// the move (Weapon.MovingSpreadPenalty), JEV stands still to shoot,
            /// and this is the one-line baseline fix anyone would try once they
            /// had seen that -- Hold dominates while in contact and loaded, and
            /// Retreat still takes over for the reload. A stronger control group,
            /// not a different one: every other weight is the basis profile's.
            /// </summary>
            public static Weights StopToShoot(Weights basis)
            {
                var w = (Weights)basis.MemberwiseClone();
                w.ContactHoldBonus = 2f;
                return w;
            }
        }

        private readonly Weights _w;
        private readonly List<ScoredIntent> _scores = new List<ScoredIntent>(TacticalIntents.Count);

        private TacticalIntent _current = TacticalIntent.Hold;
        private float _committedAt = float.NegativeInfinity;

        public UtilityBrain(Weights weights = null) => _w = weights ?? new Weights();

        public IReadOnlyList<ScoredIntent> LastScores => _scores;

        public void Reset()
        {
            _current = TacticalIntent.Hold;
            _committedAt = float.NegativeInfinity;
            _scores.Clear();
        }

        public TacticalIntent Decide(TacticalObservation obs)
        {
            _scores.Clear();

            var s = obs.Summary;
            var self = obs.Self;

            bool hurt = self.Health01 < _w.LowHealthThreshold;
            bool dry = self.Ammo01 < 0.25f || self.IsReloading > 0.5f;
            bool exposed = s.ThreatExposureHere > 0.5f;
            bool knowsEnemy = s.EnemyBeliefMass > 0.15f;
            bool inContact = self.HasLineOfSight > 0.5f;
            bool lost = self.TimeSinceEnemySeen01 > 0.6f;

            float aggression = Mathf.Clamp01(
                0.5f + _w.AggressionBias
                + (self.Health01 - 0.5f) * 0.5f
                + (self.TeamScore01 - self.EnemyScore01) * 0.2f);

            // Hold -- good when already sheltered with a firing angle.
            Score(TacticalIntent.Hold,
                0.30f
                + s.CoverQualityHere * 0.8f
                - s.ThreatExposureHere * _w.ExposurePenalty * 0.5f
                + (inContact && !dry ? _w.ContactHoldBonus : 0f)
                - (lost ? 0.3f : 0f),
                s.CoverQualityHere > 0.6f ? "sheltered" : null);

            // Retreat -- dominant when hurt, reloading, or caught in the open.
            Score(TacticalIntent.Retreat,
                (hurt ? 0.9f : 0f)
                + (dry && inContact ? 0.7f : 0f)
                + (exposed ? s.ThreatExposureHere * _w.ExposurePenalty : 0f)
                - aggression * 0.6f
                - s.CoverQualityHere * 0.5f,
                hurt ? "low hp" : dry ? "reloading" : null);

            // Push to forward cover -- advance only if somewhere better exists.
            float coverGain = s.HasBestCover ? s.BestCoverQuality - s.CoverQualityHere : -1f;
            Score(TacticalIntent.PushCoverForward,
                coverGain * _w.CoverSeekUrgency
                + aggression * 0.5f
                - DistanceCost(s.BestCoverDistance)
                - (hurt ? 0.8f : 0f),
                coverGain > 0.25f ? "better cover ahead" : null);

            // Push via off-axis cover.
            Score(TacticalIntent.PushCoverFlank,
                (s.HasFlankCover ? s.FlankCoverQuality * _w.CoverSeekUrgency : -1f)
                + aggression * 0.45f
                + (exposed ? 0.4f : 0f)
                - (hurt ? 0.6f : 0f),
                s.HasFlankCover ? "off-axis approach" : null);

            // Flanking -- worth it when we roughly know where they are but are stuck.
            float flankBase =
                (knowsEnemy ? 0.5f : -0.4f)
                + aggression * 0.4f
                + (inContact && s.EnemyBeliefDistance > 12f ? 0.3f : 0f)
                - (hurt ? 0.7f : 0f)
                - s.ThreatExposureHere * 0.5f;

            float lateral = LateralBias(obs);
            Score(TacticalIntent.FlankLeft, flankBase + lateral, null);
            Score(TacticalIntent.FlankRight, flankBase - lateral, null);

            Score(TacticalIntent.Suppress,
                (inContact && !dry ? 0.8f : -0.5f)
                + s.CoverQualityHere * 0.4f
                + (s.HasAlly && s.NearestAllyDistance < 15f ? 0.4f : 0f),
                inContact ? "pin them" : null);

            // Peek -- cheap information when safe but blind.
            Score(TacticalIntent.Peek,
                (lost ? _w.LostContactCuriosity + 0.4f : -0.2f)
                + s.CoverQualityHere * 0.5f
                + (s.EnemyBeliefSpread > 0.5f ? 0.4f : 0f)
                - (dry ? 0.3f : 0f),
                lost ? "reacquire" : null);

            Score(TacticalIntent.RegroupAlly,
                (s.HasAlly ? _w.RegroupValue : -1f)
                + (hurt && s.HasAlly ? 0.5f : 0f)
                - (s.NearestAllyDistance < 6f ? 0.8f : 0f)
                - aggression * 0.2f,
                null);

            Score(TacticalIntent.ContestCenter,
                _w.CenterValue
                + aggression * 0.5f
                - s.CenterPressure * 1.1f
                - DistanceCost(s.DistanceToCenter) * 0.6f
                - (hurt ? 0.9f : 0f)
                - (self.InCenterZone > 0.5f ? 0.5f : 0f),
                null);

            return Commit(obs);
        }

        private void Score(TacticalIntent intent, float value, string reason)
            => _scores.Add(new ScoredIntent(intent, value, reason));

        /// <summary>Mild penalty for walking a long way; not a hard cutoff.</summary>
        private static float DistanceCost(float metres) => Mathf.Clamp01(metres / 25f) * 0.7f;

        /// <summary>
        /// Positive when the left side is the softer approach. Sampled from the
        /// threat-exposure channel rather than guessed.
        /// </summary>
        private static float LateralBias(TacticalObservation obs)
        {
            const int c = TacticalObservation.CenterCell;
            float left = 0f, right = 0f;

            for (int y = c - 6; y <= c + 6; y++)
            {
                for (int d = 1; d <= 6; d++)
                {
                    if (TacticalObservation.InBounds(c - d, y))
                        left += obs.Get(ObservationChannel.ThreatExposure, c - d, y);
                    if (TacticalObservation.InBounds(c + d, y))
                        right += obs.Get(ObservationChannel.ThreatExposure, c + d, y);
                }
            }

            return Mathf.Clamp((right - left) * 0.02f, -0.5f, 0.5f);
        }

        /// <summary>
        /// Hysteresis. Without this a utility AI re-evaluates from scratch every
        /// tick and visibly twitches between two near-tied options -- the single
        /// most common reason hand-authored NPCs read as broken.
        /// </summary>
        private TacticalIntent Commit(TacticalObservation obs)
        {
            var best = TacticalIntent.Hold;
            float bestScore = float.NegativeInfinity;

            for (int i = 0; i < _scores.Count; i++)
            {
                float score = _scores[i].Score;
                if (_scores[i].Intent == _current) score += _w.CommitmentBonus;
                if (score <= bestScore) continue;
                bestScore = score;
                best = _scores[i].Intent;
            }

            if (best != _current && obs.Timestamp - _committedAt < _w.MinCommitSeconds)
                return _current;

            if (best != _current)
            {
                _current = best;
                _committedAt = obs.Timestamp;
            }

            return _current;
        }
    }
}
