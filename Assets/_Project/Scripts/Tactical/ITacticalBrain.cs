using System.Collections.Generic;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;

    /// <summary>
    /// The one seam the whole experiment turns on. Swap the implementation behind
    /// this interface and nothing else about the agent changes: same sensors, same
    /// action space, same reflex layer, same tick rate.
    /// </summary>
    public interface ITacticalBrain
    {
        /// <summary>Shown on the spectator overlay, e.g. "Utility" or "JEV".</summary>
        string Label { get; }

        /// <summary>Tactical decision for this tick.</summary>
        TacticalIntent Decide(TacticalObservation observation);

        /// <summary>
        /// Why it decided that. Populated every tick for the overlay and the
        /// decision log -- this is not debug-only scaffolding, it is the visible
        /// payload of the comparison.
        /// </summary>
        IReadOnlyList<ScoredIntent> LastScores { get; }

        void Reset();
    }

    public readonly struct ScoredIntent
    {
        public readonly TacticalIntent Intent;
        public readonly float Score;
        public readonly string Reason;

        public ScoredIntent(TacticalIntent intent, float score, string reason = null)
        {
            Intent = intent;
            Score = score;
            Reason = reason;
        }
    }
}
