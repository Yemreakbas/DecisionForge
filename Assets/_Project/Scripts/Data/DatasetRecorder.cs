using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace JevNpcBrain.Data
{
    using Arena;
    using Core;
    using Match;
    using Perception;
    using Tactical;

    /// <summary>
    /// Phase 2: records what every agent saw and what it chose, tick by tick, as
    /// the training set for the JEV world model.
    ///
    /// An episode is one agent's life in one round. Its steps are stored
    /// contiguously, so a transition (obs, intent, next obs) is just steps t and
    /// t+1 of one episode -- no observation is stored twice, and the multi-step
    /// rollouts the predictor trains on come for free.
    ///
    /// Output is a run folder: run.json plus NumPy .npz shards, so the Python side
    /// is a bare np.load. Per step: grid (float16, N x 7 x 32 x 32), self (N x 24),
    /// intent (what was executed), policy_intent (what the baseline wanted),
    /// explored, time, and pos_xz, which is for plots only and must never reach a
    /// model -- the brains are not allowed to know where they are. Per episode:
    /// start, length, agent, team, captain, round, how it ended, who won.
    ///
    /// The grid is float16 rather than bytes because three channels accumulate
    /// past 1: the noise of a burst of gunfire reaches ~10. Half precision is far
    /// below the resolution the channels themselves carry.
    ///
    /// Ticks between rounds are dropped -- the survivor wandering after the kill is
    /// not part of any fight -- and every episode closes before the respawn
    /// teleport, so no transition ever spans one.
    /// </summary>
    public sealed class DatasetRecorder : MonoBehaviour
    {
        public const string Schema = "jev-dataset/1";

        [Tooltip("Receives one folder per run. Empty = <project>/Datasets, or next to the exe in a build.")]
        public string OutputRoot;

        [Tooltip("A shard is written at the first round boundary after this many steps.")]
        public int ShardSteps = 4096;

        public string RunFolder { get; private set; }
        public int StepsWritten { get; private set; }
        public int EpisodesWritten { get; private set; }
        public int ShardsWritten { get; private set; }

        private enum EndReason : byte { Survived = 0, Died = 1, Cut = 2 }

        private struct Episode
        {
            public long Start;
            public int Length;
            public byte Agent;
            public byte Team;
            public byte Captain;
            public int Round;
            public EndReason End;
            public sbyte Won;
        }

        private sealed class OpenEpisode
        {
            public readonly StepColumns Steps = new StepColumns(256);
            public int Round;
        }

        private readonly Dictionary<NpcAgent, OpenEpisode> _open = new Dictionary<NpcAgent, OpenEpisode>();
        private readonly Stack<OpenEpisode> _pool = new Stack<OpenEpisode>();
        private readonly List<NpcAgent> _closing = new List<NpcAgent>();
        private readonly List<Episode> _episodes = new List<Episode>();

        private MatchDirector _match;
        private StepColumns _shard;
        private int _round = -1;
        private bool _roundLive;
        private int _roundsCompleted;
        private bool _begun;
        private bool _ended;
        private DateTime _startedUtc;

        public void Begin(MatchDirector match)
        {
            if (_begun) return;

            if (SelfState.FieldNames.Length != SelfState.Length)
            {
                Debug.LogError("[DatasetRecorder] SelfState.FieldNames is out of step with SelfState.Length. " +
                               "Not recording: the run.json would mislabel every feature.");
                enabled = false;
                return;
            }

            _begun = true;
            _match = match;
            _startedUtc = DateTime.UtcNow;
            _shard = new StepColumns(ShardSteps + 1024);

            string root = string.IsNullOrEmpty(OutputRoot)
                ? Path.Combine(Application.dataPath, "..", "Datasets")
                : OutputRoot;

            string name = string.Format(CultureInfo.InvariantCulture, "{0}_{1}-vs-{2}_{3:yyyyMMdd-HHmmss}_seed{4}",
                match.Format, match.TeamABrain, match.TeamBBrain, DateTime.Now, match.Seed);

            RunFolder = Path.GetFullPath(Path.Combine(root, name));
            Directory.CreateDirectory(RunFolder);

            NpcAgent.Decided += OnDecided;
            match.RoundStarted += OnRoundStarted;
            match.RoundEnded += OnRoundEnded;
            match.AgentKilled += OnAgentKilled;

            WriteRunJson(false);
            Debug.Log("[DatasetRecorder] Recording to " + RunFolder);
        }

        /// <summary>Closes whatever is still open and writes the last shard. Safe to call twice.</summary>
        public void End()
        {
            if (!_begun || _ended) return;
            _ended = true;
            _roundLive = false;

            CloseAll(EndReason.Cut);
            Flush();

            NpcAgent.Decided -= OnDecided;
            if (_match != null)
            {
                _match.RoundStarted -= OnRoundStarted;
                _match.RoundEnded -= OnRoundEnded;
                _match.AgentKilled -= OnAgentKilled;
            }

            WriteRunJson(true);
            Debug.Log(string.Format(CultureInfo.InvariantCulture,
                "[DatasetRecorder] {0} rounds, {1} episodes, {2} steps in {3} shard(s): {4}",
                _roundsCompleted, EpisodesWritten, StepsWritten, ShardsWritten, RunFolder));
        }

        /// <summary>Leaving play mode mid-run still saves everything recorded so far.</summary>
        private void OnDestroy() => End();

        private void OnRoundStarted(int round)
        {
            CloseAll(EndReason.Cut);
            _round = round;
            _roundLive = true;
        }

        private void OnDecided(NpcAgent agent)
        {
            if (!_roundLive || !agent.IsAlive) return;

            if (!_open.TryGetValue(agent, out var episode))
            {
                episode = _pool.Count > 0 ? _pool.Pop() : new OpenEpisode();
                episode.Round = _round;
                _open.Add(agent, episode);
            }

            byte intent = (byte)agent.CurrentIntent;
            byte policy = intent;
            bool explored = false;

            if (agent.Brain is ExploringBrain exploring)
            {
                policy = (byte)exploring.PolicyIntent;
                explored = exploring.Exploring;
            }

            var obs = agent.Observation;
            var position = agent.transform.position;
            episode.Steps.Add(obs.Grid, obs.SelfVector, intent, policy, explored,
                obs.Timestamp, position.x, position.z);
        }

        private void OnAgentKilled(NpcAgent victim, NpcAgent killer) => Close(victim, EndReason.Died);

        private void OnRoundEnded(int round, int winner)
        {
            _roundLive = false;
            CloseAll(EndReason.Survived);

            // Stamp the outcome on every episode of the round, including the ones
            // that ended in death before the round did.
            for (int i = _episodes.Count - 1; i >= 0 && _episodes[i].Round == round; i--)
            {
                var episode = _episodes[i];
                episode.Won = (sbyte)(winner < 0 ? -1 : episode.Team == winner ? 1 : 0);
                _episodes[i] = episode;
            }

            _roundsCompleted++;
            if (_shard.Count >= ShardSteps) Flush();
        }

        private void Close(NpcAgent agent, EndReason end)
        {
            if (agent == null || !_open.TryGetValue(agent, out var episode)) return;
            _open.Remove(agent);

            if (episode.Steps.Count > 0)
            {
                _episodes.Add(new Episode
                {
                    Start = _shard.Count,
                    Length = episode.Steps.Count,
                    Agent = (byte)agent.AgentId,
                    Team = (byte)agent.Team,
                    Captain = (byte)(agent.IsCaptain ? 1 : 0),
                    Round = episode.Round,
                    End = end,
                    Won = -1
                });

                _shard.AddRange(episode.Steps);
            }

            episode.Steps.Clear();
            _pool.Push(episode);
        }

        private void CloseAll(EndReason end)
        {
            _closing.Clear();
            _closing.AddRange(_open.Keys);
            for (int i = 0; i < _closing.Count; i++) Close(_closing[i], end);
        }

        private void Flush()
        {
            if (_shard == null || _shard.Count == 0) return;

            int n = _shard.Count;
            int e = _episodes.Count;
            string path = Path.Combine(RunFolder,
                string.Format(CultureInfo.InvariantCulture, "shard_{0:0000}.npz", ShardsWritten));

            var start = new long[e];
            var length = new int[e];
            var agent = new byte[e];
            var team = new byte[e];
            var captain = new byte[e];
            var round = new int[e];
            var end = new byte[e];
            var won = new sbyte[e];

            for (int i = 0; i < e; i++)
            {
                var episode = _episodes[i];
                start[i] = episode.Start;
                length[i] = episode.Length;
                agent[i] = episode.Agent;
                team[i] = episode.Team;
                captain[i] = episode.Captain;
                round[i] = episode.Round;
                end[i] = (byte)episode.End;
                won[i] = episode.Won;
            }

            try
            {
                using (var npz = new NpzWriter(path))
                {
                    npz.WriteHalf("grid", _shard.Grid, n, TacticalObservation.ChannelCount,
                        TacticalObservation.GridSize, TacticalObservation.GridSize);
                    npz.Write("self", _shard.Self, n, SelfState.Length);
                    npz.Write("intent", _shard.Intent, n);
                    npz.Write("policy_intent", _shard.Policy, n);
                    npz.Write("explored", _shard.Explored, n);
                    npz.Write("time", _shard.Time, n);
                    npz.Write("pos_xz", _shard.PosXZ, n, 2);

                    npz.Write("ep_start", start, e);
                    npz.Write("ep_length", length, e);
                    npz.Write("ep_agent", agent, e);
                    npz.Write("ep_team", team, e);
                    npz.Write("ep_captain", captain, e);
                    npz.Write("ep_round", round, e);
                    npz.Write("ep_end", end, e);
                    npz.Write("ep_won", won, e);
                }
            }
            catch (Exception exception)
            {
                Debug.LogError("[DatasetRecorder] Could not write " + path + ": " + exception.Message);
                return;
            }

            StepsWritten += n;
            EpisodesWritten += e;
            ShardsWritten++;
            _shard.Clear();
            _episodes.Clear();

            WriteRunJson(false);
        }

        /// <summary>
        /// Everything the trainer needs to interpret the arrays. Rewritten after
        /// every shard, so a crashed run still describes what it did write.
        /// Numbers go through InvariantCulture: this machine is Turkish-locale, and
        /// "0,1" in a JSON file is a syntax error at best.
        /// </summary>
        private void WriteRunJson(bool complete)
        {
            var c = CultureInfo.InvariantCulture;
            var layout = ArenaLayout.Current;
            float tickHz = NpcAgent.All.Count > 0 ? NpcAgent.All[0].TacticalHz : 10f;

            var json = new StringBuilder(2048);
            json.Append("{\n");
            json.Append("  \"schema\": ").Append(Quote(Schema)).Append(",\n");
            json.Append("  \"created_utc\": ").Append(Quote(_startedUtc.ToString("yyyy-MM-ddTHH:mm:ssZ", c))).Append(",\n");
            json.Append("  \"unity_version\": ").Append(Quote(Application.unityVersion)).Append(",\n");
            json.Append("  \"arena\": ").Append(Quote(layout != null ? layout.Name : "?")).Append(",\n");
            json.Append("  \"format\": ").Append(Quote(_match.Format.ToString())).Append(",\n");
            json.Append("  \"brains\": [").Append(Quote(_match.TeamABrain.ToString())).Append(", ")
                .Append(Quote(_match.TeamBBrain.ToString())).Append("],\n");
            json.Append("  \"seed\": ").Append(_match.Seed.ToString(c)).Append(",\n");
            json.Append("  \"tick_hz\": ").Append(tickHz.ToString("R", c)).Append(",\n");
            json.Append("  \"sim_step_seconds\": ").Append(Time.captureDeltaTime.ToString("R", c)).Append(",\n");
            json.Append("  \"exploration\": {\"start_chance_per_decision\": ")
                .Append(_match.ExplorationRate.ToString("R", c))
                .Append(", \"min_seconds\": ").Append(_match.ExploreMinSeconds.ToString("R", c))
                .Append(", \"max_seconds\": ").Append(_match.ExploreMaxSeconds.ToString("R", c)).Append("},\n");
            json.Append("  \"grid\": {\"channels\": ").Append(NameList(typeof(ObservationChannel)))
                .Append(", \"size\": ").Append(TacticalObservation.GridSize.ToString(c))
                .Append(", \"cell_size\": ").Append(TacticalObservation.CellSize.ToString("R", c))
                .Append(", \"layout\": \"CHW\", \"frame\": \"agent-centred, world-axis-aligned\"")
                .Append(", \"dtype\": \"float16\"},\n");
            json.Append("  \"self_fields\": ").Append(List(SelfState.FieldNames)).Append(",\n");
            json.Append("  \"intents\": ").Append(NameList(typeof(TacticalIntent))).Append(",\n");
            json.Append("  \"episode_end\": [\"survived\", \"died\", \"cut\"],\n");
            json.Append("  \"debug_only\": [\"pos_xz\"],\n");
            json.Append("  \"complete\": ").Append(complete ? "true" : "false").Append(",\n");
            json.Append("  \"totals\": {\"rounds\": ").Append(_roundsCompleted.ToString(c))
                .Append(", \"shards\": ").Append(ShardsWritten.ToString(c))
                .Append(", \"episodes\": ").Append(EpisodesWritten.ToString(c))
                .Append(", \"steps\": ").Append(StepsWritten.ToString(c)).Append("}\n");
            json.Append("}\n");

            File.WriteAllText(Path.Combine(RunFolder, "run.json"), json.ToString(), new UTF8Encoding(false));
        }

        /// <summary>
        /// Enum names in value order, which is the order the model sees them in.
        /// GetNames is documented to sort by value, and both enums start at zero.
        /// </summary>
        private static string NameList(Type enumType) => List(Enum.GetNames(enumType));

        private static string List(string[] items)
        {
            var parts = new string[items.Length];
            for (int i = 0; i < items.Length; i++) parts[i] = Quote(items[i]);
            return "[" + string.Join(", ", parts) + "]";
        }

        private static string Quote(string text) => "\"" + text.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    /// <summary>One growable array per recorded field, in step order.</summary>
    internal sealed class StepColumns
    {
        private const int GridLength = TacticalObservation.GridLength;
        private const int SelfLength = SelfState.Length;

        public int Count;
        public ushort[] Grid;
        public float[] Self;
        public byte[] Intent;
        public byte[] Policy;
        public byte[] Explored;
        public float[] Time;
        public float[] PosXZ;

        private int _capacity;

        public StepColumns(int capacity)
        {
            _capacity = Mathf.Max(capacity, 16);
            Grid = new ushort[_capacity * GridLength];
            Self = new float[_capacity * SelfLength];
            Intent = new byte[_capacity];
            Policy = new byte[_capacity];
            Explored = new byte[_capacity];
            Time = new float[_capacity];
            PosXZ = new float[_capacity * 2];
        }

        public void Clear() => Count = 0;

        public void Add(float[] grid, float[] self, byte intent, byte policy, bool explored,
            float time, float x, float z)
        {
            Reserve(Count + 1);

            int g = Count * GridLength;
            for (int i = 0; i < GridLength; i++) Grid[g + i] = Half16.FromFloat(grid[i]);

            Array.Copy(self, 0, Self, Count * SelfLength, SelfLength);
            Intent[Count] = intent;
            Policy[Count] = policy;
            Explored[Count] = explored ? (byte)1 : (byte)0;
            Time[Count] = time;
            PosXZ[Count * 2] = x;
            PosXZ[Count * 2 + 1] = z;
            Count++;
        }

        public void AddRange(StepColumns other)
        {
            Reserve(Count + other.Count);

            Array.Copy(other.Grid, 0, Grid, Count * GridLength, other.Count * GridLength);
            Array.Copy(other.Self, 0, Self, Count * SelfLength, other.Count * SelfLength);
            Array.Copy(other.Intent, 0, Intent, Count, other.Count);
            Array.Copy(other.Policy, 0, Policy, Count, other.Count);
            Array.Copy(other.Explored, 0, Explored, Count, other.Count);
            Array.Copy(other.Time, 0, Time, Count, other.Count);
            Array.Copy(other.PosXZ, 0, PosXZ, Count * 2, other.Count * 2);
            Count += other.Count;
        }

        private void Reserve(int needed)
        {
            if (needed <= _capacity) return;

            int capacity = Mathf.Max(needed, _capacity * 2);
            Array.Resize(ref Grid, capacity * GridLength);
            Array.Resize(ref Self, capacity * SelfLength);
            Array.Resize(ref Intent, capacity);
            Array.Resize(ref Policy, capacity);
            Array.Resize(ref Explored, capacity);
            Array.Resize(ref Time, capacity);
            Array.Resize(ref PosXZ, capacity * 2);
            _capacity = capacity;
        }
    }
}
