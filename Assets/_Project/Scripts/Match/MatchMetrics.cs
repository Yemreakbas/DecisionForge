using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace JevNpcBrain.Match
{
    using Core;
    using Perception;

    /// <summary>
    /// Per-team tallies for one experiment run.
    ///
    /// Win rate alone would let a worse-feeling AI look better on paper, so the
    /// interesting columns here are the qualitative ones: how often an agent
    /// froze in the open, how much time it spent exposed per kill, and who won
    /// first contact. Those are what a player actually perceives as intelligence.
    /// </summary>
    public sealed class TeamStats
    {
        public string BrainLabel = "?";
        public int RoundsWon;
        public int Kills;
        public int Deaths;
        public int FirstContactWins;
        public int DumbMoments;
        public float ExposureSeconds;

        /// <summary>Seconds any living member held the contested centre.</summary>
        public float CenterSeconds;
        public float DecisionMillisecondsTotal;
        public int DecisionSamples;

        public float AverageDecisionMs => DecisionSamples > 0
            ? DecisionMillisecondsTotal / DecisionSamples
            : 0f;

        public float PeakDecisionMs;

        public void Reset()
        {
            RoundsWon = Kills = Deaths = FirstContactWins = DumbMoments = 0;
            ExposureSeconds = CenterSeconds = DecisionMillisecondsTotal = PeakDecisionMs = 0f;
            DecisionSamples = 0;
        }
    }

    /// <summary>
    /// Samples the live agents and turns their state into the numbers the
    /// experiment reports. Runs at the tactical rate, not per frame -- the metrics
    /// must not cost more than the thing they measure.
    /// </summary>
    public sealed class MatchMetrics : MonoBehaviour
    {
        [Tooltip("Seconds in the open, motionless and exposed, before it counts as a dumb moment.")]
        public float FrozenInTheOpenSeconds = 1.5f;

        public float SampleHz = 10f;

        public readonly TeamStats[] Teams = { new TeamStats(), new TeamStats() };

        private readonly Dictionary<int, float> _frozenTimers = new Dictionary<int, float>();
        private readonly Dictionary<int, float> _dumbCooldowns = new Dictionary<int, float>();

        private float _timer;

        /// <summary>Rounds that ended with nobody winning. A high count means the
        /// arena is not forcing contact, which makes the data as well as the
        /// footage worthless.</summary>
        public int RoundsDrawn;

        public void ResetAll()
        {
            RoundsDrawn = 0;
            Teams[0].Reset();
            Teams[1].Reset();
            _frozenTimers.Clear();
            _dumbCooldowns.Clear();
        }

        public void RecordKill(int killerTeam, int victimTeam)
        {
            if (killerTeam >= 0 && killerTeam < 2) Teams[killerTeam].Kills++;
            if (victimTeam >= 0 && victimTeam < 2) Teams[victimTeam].Deaths++;
        }

        public void RecordRoundWin(int team)
        {
            if (team >= 0 && team < 2) Teams[team].RoundsWon++;
        }

        public void RecordFirstContact(int team)
        {
            if (team >= 0 && team < 2) Teams[team].FirstContactWins++;
        }

        private void Update()
        {
            float interval = 1f / Mathf.Max(SampleHz, 1f);
            _timer -= Time.deltaTime;
            if (_timer > 0f) return;
            _timer += interval;

            Sample(interval);
        }

        private void Sample(float dt)
        {
            for (int i = 0; i < NpcAgent.All.Count; i++)
            {
                var agent = NpcAgent.All[i];
                if (!agent.IsAlive || agent.Team < 0 || agent.Team > 1) continue;

                var stats = Teams[agent.Team];
                var summary = agent.Observation.Summary;

                stats.DecisionMillisecondsTotal += agent.LastDecisionMilliseconds;
                stats.DecisionSamples++;
                if (agent.LastDecisionMilliseconds > stats.PeakDecisionMs)
                    stats.PeakDecisionMs = agent.LastDecisionMilliseconds;

                bool exposed = summary.ThreatExposureHere > 0.5f;
                if (exposed) stats.ExposureSeconds += dt;

                if (agent.Observation.Self.InCenterZone > 0.5f) stats.CenterSeconds += dt;

                TrackDumbMoments(agent, stats, summary, exposed, dt);
            }
        }

        /// <summary>
        /// Two failure modes a player reads instantly as a broken NPC: standing
        /// still in the open while someone can shoot you, and walking into
        /// geometry. Both are counted as discrete events with a cooldown, so one
        /// long mistake does not inflate the tally into meaninglessness.
        /// </summary>
        private void TrackDumbMoments(NpcAgent agent, TeamStats stats,
            ObservationSummary summary, bool exposed, float dt)
        {
            int id = agent.AgentId;

            _dumbCooldowns.TryGetValue(id, out float cooldown);
            cooldown = Mathf.Max(0f, cooldown - dt);

            _frozenTimers.TryGetValue(id, out float frozen);

            bool motionless = agent.Motor != null && agent.Motor.Speed01 < 0.15f;
            bool uncovered = summary.CoverQualityHere < 0.15f;
            frozen = exposed && motionless && uncovered ? frozen + dt : 0f;

            bool stuck = agent.Motor != null && agent.Motor.IsStuck;

            if (cooldown <= 0f && (frozen >= FrozenInTheOpenSeconds || stuck))
            {
                stats.DumbMoments++;
                cooldown = 2f;
                frozen = 0f;
            }

            _frozenTimers[id] = frozen;
            _dumbCooldowns[id] = cooldown;
        }

        /// <summary>
        /// Invariant culture, always. On a Turkish-locale machine the default
        /// formatter writes "111,8", which silently splits a column and hands the
        /// phase 2 Python trainer a quietly wrong dataset.
        /// </summary>
        public string ToCsv()
        {
            var c = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine("team,brain,rounds_won,rounds_drawn,kills,deaths,first_contact," +
                          "dumb_moments,exposure_s,center_s,avg_decision_ms,peak_decision_ms");

            for (int t = 0; t < 2; t++)
            {
                var s = Teams[t];
                sb.AppendLine(string.Format(c,
                    "{0},{1},{2},{3},{4},{5},{6},{7},{8:F1},{9:F1},{10:F4},{11:F4}",
                    t, s.BrainLabel, s.RoundsWon, RoundsDrawn, s.Kills, s.Deaths,
                    s.FirstContactWins, s.DumbMoments, s.ExposureSeconds, s.CenterSeconds,
                    s.AverageDecisionMs, s.PeakDecisionMs));
            }

            return sb.ToString();
        }
    }
}
