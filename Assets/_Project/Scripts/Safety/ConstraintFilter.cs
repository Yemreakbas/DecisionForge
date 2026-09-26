using System.Collections.Generic;

namespace JevNpcBrain.Safety
{
    using Core;
    using Perception;
    using Tactical;

    /// <summary>
    /// The SAFETY layer: a hard-rule veto over whatever the tactical layer
    /// proposed. It sits between <see cref="ITacticalBrain.Decide"/> and the reflex
    /// layer and applies to every brain alike -- it is part of the shared body, not
    /// of either mind.
    ///
    /// Every rule here vetoes something that is pointless or suicidal *by
    /// construction*, not something a designer merely dislikes. Each was checked
    /// against 119,151 recorded baseline decisions (the calibration runs plus seed
    /// 1004) and fires on none of them, so for UtilityBrain the filter is a no-op
    /// and the 19-21 calibration still describes the build. A rule that would
    /// have overridden the baseline -- "no advancing while reloading in contact",
    /// 1.8% of its decisions -- was left out for exactly that reason.
    ///
    /// A learned planner is what these rules are for: it can find a gap in its
    /// own world model and pour intent into it.
    /// </summary>
    public static class ConstraintFilter
    {
        public const float CriticalHealth = 0.25f;
        public const float ExposedThreshold = 0.5f;

        /// <summary>Why <paramref name="intent"/> is vetoed right now, or null if it is allowed.</summary>
        public static string Veto(TacticalIntent intent, TacticalObservation obs)
        {
            var self = obs.Self;
            switch (intent)
            {
                case TacticalIntent.RegroupAlly:
                    if (self.AlliesAlive01 <= 0f) return "no allies";
                    break;

                case TacticalIntent.Suppress:
                    if (self.Ammo01 <= 0.001f || self.IsReloading > 0.5f) return "nothing to fire";
                    break;

                case TacticalIntent.PushCoverForward:
                case TacticalIntent.ContestCenter:
                    if (self.Health01 < CriticalHealth && obs.Summary.ThreatExposureHere > ExposedThreshold)
                        return "critical and exposed";
                    break;
            }

            // Hold is never vetoed: it is the fallback when everything else is.
            return null;
        }

        public static bool IsAllowed(TacticalIntent intent, TacticalObservation obs) => Veto(intent, obs) == null;

        /// <summary>
        /// Returns <paramref name="proposed"/> if it is allowed. Otherwise the brain's
        /// best-scored allowed alternative, so a veto degrades to the brain's own
        /// second choice rather than to a rule-writer's guess; Hold if it has none.
        /// </summary>
        public static TacticalIntent Apply(TacticalIntent proposed, TacticalObservation obs,
            IReadOnlyList<ScoredIntent> scores, out string reason)
        {
            reason = Veto(proposed, obs);
            if (reason == null) return proposed;

            var best = TacticalIntent.Hold;
            float bestScore = float.NegativeInfinity;
            if (scores != null)
            {
                for (int i = 0; i < scores.Count; i++)
                {
                    var candidate = scores[i];
                    if (candidate.Intent == proposed || candidate.Score <= bestScore) continue;
                    if (!IsAllowed(candidate.Intent, obs)) continue;
                    bestScore = candidate.Score;
                    best = candidate.Intent;
                }
            }

            reason = $"{TacticalIntents.Label(proposed)} vetoed: {reason}";
            return best;
        }
    }
}
