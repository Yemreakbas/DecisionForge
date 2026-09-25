using System.Collections.Generic;

namespace JevNpcBrain.Tactical
{
    using Core;
    using Perception;

    /// <summary>
    /// The data-collection policy: the baseline, with occasional committed detours.
    ///
    /// Phase 2 records what follows each intent, and a world model can only learn
    /// the consequences of actions it has actually seen. The baseline alone never
    /// picks ContestCenter -- PushCoverForward always outbids it -- and flanks
    /// rarely. So now and then this takes over and commits to a random intent for
    /// a second or three. Committed rather than re-rolled each tick: a fresh
    /// random choice ten times a second is jitter, and what follows it is noise.
    ///
    /// Collection runs only, and always both teams or neither. Exploring makes
    /// the baseline worse on purpose, so no evaluation match may ever use this.
    /// </summary>
    public sealed class ExploringBrain : ITacticalBrain
    {
        private readonly ITacticalBrain _inner;
        private readonly System.Random _random;
        private readonly float _startChance;
        private readonly float _minSeconds;
        private readonly float _maxSeconds;

        private TacticalIntent _detour;
        private float _detourUntil = float.NegativeInfinity;

        /// <param name="startChance">Chance per decision of starting a detour.</param>
        public ExploringBrain(ITacticalBrain inner, float startChance,
            float minSeconds, float maxSeconds, int seed)
        {
            _inner = inner;
            _startChance = startChance;
            _minSeconds = minSeconds;
            _maxSeconds = maxSeconds;
            _random = new System.Random(seed);
        }

        public string Label => _inner.Label;

        /// <summary>The baseline's own scores, even while a detour overrides them.</summary>
        public IReadOnlyList<ScoredIntent> LastScores => _inner.LastScores;

        /// <summary>What the baseline wanted this tick, before any detour.</summary>
        public TacticalIntent PolicyIntent { get; private set; }

        /// <summary>True when this tick's intent came from a detour.</summary>
        public bool Exploring { get; private set; }

        public TacticalIntent Decide(TacticalObservation observation)
        {
            // The baseline decides every tick regardless, so its hysteresis state
            // stays honest and the dataset records what it would have done.
            PolicyIntent = _inner.Decide(observation);

            float now = observation.Timestamp;
            if (now >= _detourUntil && _random.NextDouble() < _startChance)
            {
                _detour = TacticalIntents.All[_random.Next(TacticalIntents.Count)];
                _detourUntil = now + _minSeconds + (float)_random.NextDouble() * (_maxSeconds - _minSeconds);
            }

            Exploring = now < _detourUntil;
            return Exploring ? _detour : PolicyIntent;
        }

        public void Reset()
        {
            _inner.Reset();
            _detourUntil = float.NegativeInfinity;
            Exploring = false;
        }
    }
}
