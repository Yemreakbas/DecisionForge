using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace JevNpcBrain.Match
{
    using Arena;
    using Combat;
    using Core;
    using Data;
    using Perception;
    using Presentation;
    using Reflex;
    using Tactical;

    /// <summary>Serialized by value in the scene: append only.</summary>
    public enum BrainKind { Utility, Jev, UtilityStopToShoot }

    public enum MatchFormat { CaptainDuel, Squad }

    /// <summary>
    /// Runs the experiment: spawns two teams, plays rounds, swaps spawns, and
    /// writes the numbers out.
    ///
    /// The spawn swap every round is not cosmetic. Even a point-symmetric arena
    /// can favour a side through spawn facing or who reaches the centre first, and
    /// without the swap that bias would be silently attributed to the brain.
    /// </summary>
    public sealed class MatchDirector : MonoBehaviour
    {
        [Header("Format")]
        public MatchFormat Format = MatchFormat.CaptainDuel;
        public BrainKind TeamABrain = BrainKind.Utility;
        public BrainKind TeamBBrain = BrainKind.Utility;

        [Tooltip("World model for BrainKind.Jev teams (Assets/_Project/Models/<checkpoint>/JevModel.asset).")]
        public JevModelAsset JevModel;

        [Tooltip("Other world models the build carries, picked by checkpoint name with " +
                 "'-jevmodel <checkpoint>'. Only referenced models make it into a player.")]
        public JevModelAsset[] AlternativeJevModels;

        [Tooltip("Draw every JEV brain's imagined futures. Presentation only.")]
        public bool ShowImagination = true;

        [Tooltip("Ablation: overrides Weapon.MovingSpreadPenalty for every agent. Negative keeps " +
                 "the weapon's own value (the real rules). CSVs of an override are tagged _spread<X>.")]
        public float MovingSpreadOverride = -1f;

        [Tooltip("Auto: GPU compute when a graphics device exists, CPU otherwise (-nographics). " +
                 "In the editor the CPU backend measured ~240 ms per decision against ~9 ms on GPU.")]
        public JevBackend JevInference = JevBackend.Auto;

        public enum JevBackend { Auto, CPU, GPUCompute }

        [Header("Look")]
        [Tooltip("Optional humanoid rig with an Animator. Empty = procedural mascot. " +
                 "See AgentVisual for the animator parameter contract.")]
        public GameObject CharacterPrefab;

        [Tooltip("Rifle model, parented to the right hand bone. Needs a 'LeftGrip' child.")]
        public GameObject WeaponPrefab;

        [Tooltip("Vault and slide-into-cover. Turn on only once the clips exist -- " +
                 "applies to both teams, so the fairness contract holds either way.")]
        public bool EnableAdvancedMovement = false;

        [Header("Simulation")]
        [Tooltip("Fast-forward for unattended calibration and data-collection runs.")]
        [Range(0.25f, 8f)] public float SimulationSpeed = 3f;

        [Header("Rounds")]
        public int Rounds = 40;
        public float RoundTimeLimit = 45f;
        public float IntermissionSeconds = 1.5f;

        [Header("Output")]
        public bool WriteCsvOnFinish = true;

        [Header("Cameras")]
        [Tooltip("Captains get a helmet cam and an over-the-shoulder camera. " +
                 "CameraDirector lists the keys.")]
        public bool CaptainCameras = true;

        [Tooltip("Give every agent cameras, not just captains. Noisy in a 3v3.")]
        public bool CamerasForEveryone = false;

        [Header("Data collection (phase 2)")]
        [Tooltip("Record every agent's observations and intents for the JEV trainer. " +
                 "Also switches on exploration and a fixed 60 Hz simulation step, and " +
                 "tags the CSV: an exploring match is never an evaluation match.")]
        public bool CollectDataset;

        [Tooltip("Chance per decision of starting a committed random detour. Both teams alike.")]
        [Range(0f, 0.1f)] public float ExplorationRate = 0.012f;

        public float ExploreMinSeconds = 0.8f;
        public float ExploreMaxSeconds = 3f;

        [Tooltip("Seeds Unity's Random and the exploration, so a run can be replayed.")]
        public int Seed = 1;

        [Tooltip("Empty = <project>/Datasets.")]
        public string DatasetFolder;

        [Header("Wiring")]
        public ArenaBuilder Arena;
        public MatchMetrics Metrics;
        public CameraDirector Cameras;
        public DatasetRecorder Recorder;

        private readonly List<NpcAgent> _agents = new List<NpcAgent>();
        private readonly Material[] _teamMaterials = new Material[2];

        private int _round;
        private bool _firstContactRecorded;
        private readonly float[] _roundCenterSeconds = new float[2];

        public int RoundIndex => _round;
        public bool IsRunning { get; private set; }

        /// <summary>Raised once a round's agents are placed. Presentation listens; the experiment does not.</summary>
        public event System.Action<int> RoundStarted;

        /// <summary>Raised on every kill with (victim, killer). The killer is null when no agent fired the shot.</summary>
        public event System.Action<NpcAgent, NpcAgent> AgentKilled;

        /// <summary>Raised when a round is decided, with (round, winning team), or -1 for no winner.</summary>
        public event System.Action<int, int> RoundEnded;

        private bool _quitWhenDone;

        /// <summary>Headless A/B run (-evaluate): fixed step like collection, no recording.</summary>
        private bool _evaluate;

        private bool FixedStep => CollectDataset || _evaluate;

        private IEnumerator Start()
        {
            ApplyCommandLine();

            // Without this the Editor throttles play mode to a crawl whenever it
            // loses focus, which silently ruins any long unattended run -- and we
            // intend to leave thousands of rounds grinding in the background.
            Application.runInBackground = true;

            if (FixedStep)
            {
                // Fixed 60 Hz steps, as fast as the machine allows. The tactical
                // tick then lands every 0.1 s of game time exactly. On a scaled,
                // variable step it runs at most once per frame, so it falls behind
                // whenever a frame covers more than 0.1 s -- and every transition
                // in the dataset would span a different interval.
                Time.timeScale = 1f;
                Time.captureDeltaTime = 1f / 60f;
                Random.InitState(Seed);
            }
            else
            {
                Time.timeScale = Mathf.Clamp(SimulationSpeed, 0.25f, 8f);
            }

            if (Arena == null) Arena = FindAnyObjectByType<ArenaBuilder>();
            if (Metrics == null) Metrics = gameObject.AddComponent<MatchMetrics>();
            if (Cameras == null) Cameras = FindAnyObjectByType<CameraDirector>();

            bool anyJev = TeamABrain == BrainKind.Jev || TeamBBrain == BrainKind.Jev;
            if (anyJev && ShowImagination && !Application.isBatchMode && GetComponent<ImaginationTrails>() == null)
                gameObject.AddComponent<ImaginationTrails>();

            // The arena builds in its own Awake; wait a frame so the layout exists.
            yield return null;

            if (ArenaLayout.Current == null)
            {
                Debug.LogError("[MatchDirector] No ArenaLayout. Is ArenaBuilder in the scene?");
                yield break;
            }

            CreateTeamMaterials();
            SpawnTeams();

            Metrics.Teams[0].BrainLabel = BrainLabel(TeamABrain);
            Metrics.Teams[1].BrainLabel = BrainLabel(TeamBBrain);
            Metrics.ResetAll();

            if (CollectDataset)
            {
                if (Recorder == null) Recorder = gameObject.AddComponent<DatasetRecorder>();
                if (!string.IsNullOrEmpty(DatasetFolder)) Recorder.OutputRoot = DatasetFolder;
                Recorder.Begin(this);
            }

            yield return RunMatch();
        }

        /// <summary>
        /// Headless collection from a player build (see CollectorBuild and
        /// Training/collect.ps1):
        ///   JevCollector.exe -batchmode -nographics -collect -rounds 500 -seed 3
        ///                    [-format squad] [-explore 0.012] [-out D:\Datasets]
        /// The editor's own command line carries none of these flags, so this is
        /// inert there. Case folding and number parsing are invariant, so a run
        /// behaves the same whatever locale the machine is set to.
        /// </summary>
        private void ApplyCommandLine()
        {
            var args = System.Environment.GetCommandLineArgs();
            var c = CultureInfo.InvariantCulture;

            for (int i = 0; i < args.Length; i++)
            {
                string value = i + 1 < args.Length ? args[i + 1] : null;

                switch (args[i].ToLowerInvariant())
                {
                    case "-collect":
                        CollectDataset = true;
                        _quitWhenDone = true;
                        Application.logMessageReceived += QuitOnException;
                        break;
                    case "-rounds":
                        if (int.TryParse(value, NumberStyles.Integer, c, out int rounds)) Rounds = rounds;
                        break;
                    case "-seed":
                        if (int.TryParse(value, NumberStyles.Integer, c, out int seed)) Seed = seed;
                        break;
                    case "-explore":
                        if (float.TryParse(value, NumberStyles.Float, c, out float rate)) ExplorationRate = rate;
                        break;
                    case "-format":
                        if (value != null)
                            Format = value.ToLowerInvariant() == "squad" ? MatchFormat.Squad : MatchFormat.CaptainDuel;
                        break;
                    case "-out":
                        if (!string.IsNullOrEmpty(value)) DatasetFolder = value;
                        break;
                    case "-evaluate":
                        _evaluate = true;
                        _quitWhenDone = true;
                        Application.logMessageReceived += QuitOnException;
                        break;
                    case "-braina":
                        if (value != null) TeamABrain = ParseBrain(value, TeamABrain);
                        break;
                    case "-brainb":
                        if (value != null) TeamBBrain = ParseBrain(value, TeamBBrain);
                        break;
                    case "-spreadpenalty":
                        if (float.TryParse(value, NumberStyles.Float, c, out float penalty)) MovingSpreadOverride = penalty;
                        break;
                    case "-jevmodel":
                        _jevModelName = value;
                        break;
                }
            }
        }

        /// <summary>
        /// A collector that hits an exception must exit, not idle forever inside
        /// collect.ps1's wait -- the first headless build did exactly that when
        /// agent spawning threw.
        /// </summary>
        private static void QuitOnException(string message, string stackTrace, LogType type)
        {
            if (type == LogType.Exception) Application.Quit(1);
        }

        private Unity.InferenceEngine.BackendType ResolveJevBackend()
        {
            bool gpu = JevInference == JevBackend.GPUCompute
                || (JevInference == JevBackend.Auto && SystemInfo.supportsComputeShaders
                    && SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Null);
            return gpu ? Unity.InferenceEngine.BackendType.GPUCompute : Unity.InferenceEngine.BackendType.CPU;
        }

        private static BrainKind ParseBrain(string value, BrainKind fallback)
        {
            switch (value.ToLowerInvariant())
            {
                case "jev": return BrainKind.Jev;
                case "utility": return BrainKind.Utility;
                case "stopshoot": return BrainKind.UtilityStopToShoot;
                default:
                    Debug.LogWarning($"[MatchDirector] Unknown brain '{value}', keeping {fallback}.");
                    return fallback;
            }
        }

        /// <summary>A Jev label names its model: two checkpoints are two different brains.</summary>
        private string BrainLabel(BrainKind kind)
        {
            string label = kind == BrainKind.Jev && _jevRuntime != null
                ? "Jev:" + _jevRuntime.Contract.checkpoint
                : kind.ToString();
            return CollectDataset && ExplorationRate > 0f ? label + "+explore" : label;
        }

        private string _jevModelName;

        /// <summary>JevModel, or the model -jevmodel names among it and the alternatives.</summary>
        private JevModelAsset SelectJevModel()
        {
            if (string.IsNullOrEmpty(_jevModelName)) return JevModel;

            var candidates = new List<JevModelAsset> { JevModel };
            if (AlternativeJevModels != null) candidates.AddRange(AlternativeJevModels);
            foreach (var model in candidates)
                if (model != null && model.LoadContract().checkpoint == _jevModelName) return model;

            throw new System.InvalidOperationException(
                $"[MatchDirector] -jevmodel '{_jevModelName}' is neither JevModel nor in AlternativeJevModels.");
        }

        /// <summary>Leaving play mode must not leave the editor stepping in fixed time.</summary>
        private void OnDestroy()
        {
            if (FixedStep) Time.captureDeltaTime = 0f;

            // Inference Engine workers hold native memory; release it with the match.
            _jevRuntime?.Dispose();
            _jevRuntime = null;
        }

        private JevRuntime _jevRuntime;

        private int AgentsPerTeam => Format == MatchFormat.CaptainDuel ? 1 : 3;

        private void CreateTeamMaterials()
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");

            _teamMaterials[0] = new Material(shader) { name = "TeamA" };
            _teamMaterials[0].color = new Color(0.25f, 0.55f, 0.95f);

            _teamMaterials[1] = new Material(shader) { name = "TeamB" };
            _teamMaterials[1].color = new Color(0.95f, 0.35f, 0.30f);
        }

        private void SpawnTeams()
        {
            for (int team = 0; team < 2; team++)
            {
                var kind = team == 0 ? TeamABrain : TeamBBrain;

                for (int i = 0; i < AgentsPerTeam; i++)
                {
                    bool captain = i == 0;
                    string name = captain
                        ? $"Team{(team == 0 ? "A" : "B")}_Captain"
                        : $"Team{(team == 0 ? "A" : "B")}_{i}";

                    int agentId = team * AgentsPerTeam + i;
                    var agent = CreateAgent(name, team, captain);
                    agent.Configure(agentId, team, captain, name, MakeBrain(kind, captain, agentId));
                    AttachCameras(agent, captain);
                    _agents.Add(agent);
                }
            }
        }

        /// <summary>
        /// A Jev team without a model has no brain to run: that is a setup error,
        /// and quietly playing the baseline under a "Jev" label would poison every
        /// number downstream, so it throws.
        /// </summary>
        private ITacticalBrain MakeBrain(BrainKind kind, bool captain, int agentId)
        {
            ITacticalBrain brain;
            if (kind == BrainKind.Jev)
            {
                if (JevModel == null)
                    throw new System.InvalidOperationException(
                        "[MatchDirector] A team is set to Jev but no JevModel is assigned.");
                if (_jevRuntime == null)
                {
                    var backend = ResolveJevBackend();
                    _jevRuntime = new JevRuntime(SelectJevModel(), backend);
                    Debug.Log($"[MatchDirector] JEV model '{_jevRuntime.Contract.checkpoint}' on {backend}.");
                }
                brain = new JevBrain(_jevRuntime, captain ? JevBrain.Weights.Captain() : null);
            }
            else
            {
                var weights = captain ? UtilityBrain.Weights.Captain() : new UtilityBrain.Weights();
                if (kind == BrainKind.UtilityStopToShoot) weights = UtilityBrain.Weights.StopToShoot(weights);
                brain = new UtilityBrain(weights);
            }

            // Collection runs only, and every agent alike -- see ExploringBrain.
            if (CollectDataset && ExplorationRate > 0f)
                brain = new ExploringBrain(brain, ExplorationRate, ExploreMinSeconds, ExploreMaxSeconds,
                    Seed * 7919 + agentId);

            return brain;
        }

        private NpcAgent CreateAgent(string name, int team, bool captain)
        {
            var root = new GameObject(name);
            root.transform.SetParent(transform, false);

            var controller = root.AddComponent<CharacterController>();
            controller.height = 1.8f;
            controller.radius = 0.35f;
            controller.center = Vector3.up * 0.9f;

            var eye = new GameObject("Eye").transform;
            eye.SetParent(root.transform, false);
            // Pushed forward so a shot never starts inside our own capsule.
            eye.localPosition = new Vector3(0f, 1.55f, 0.5f);

            var health = root.AddComponent<Damageable>();
            var weapon = eye.gameObject.AddComponent<Weapon>();
            if (MovingSpreadOverride >= 0f) weapon.MovingSpreadPenalty = MovingSpreadOverride;
            var motor = root.AddComponent<ReflexMotor>();
            motor.Eye = eye;
            motor.EnableVault = EnableAdvancedMovement;
            motor.EnableSlide = EnableAdvancedMovement;

            var agent = root.AddComponent<NpcAgent>();
            agent.Health = health;
            agent.Weapon = weapon;
            agent.Motor = motor;
            agent.Eye = eye;

            // A headless collector renders nothing, and no rule reads the body: the
            // rig has no colliders and shots leave from the Eye. Skipping it saves
            // the Animator, IK and skinning for every agent on every frame.
            if (!Application.isBatchMode)
            {
                var visual = root.AddComponent<AgentVisual>();
                visual.RigPrefab = CharacterPrefab;
                visual.WeaponPrefab = WeaponPrefab;
                visual.Build(_teamMaterials[team], captain);
            }

            health.Died += killer => OnAgentDied(agent, killer);
            health.Damaged += (attacker, _) => OnAgentDamaged(agent, attacker);

            return agent;
        }

        /// <summary>
        /// After Configure, so the rig and the director see the agent's final name,
        /// team and captaincy -- the director's number keys are assigned by them.
        /// </summary>
        private void AttachCameras(NpcAgent agent, bool captain)
        {
            // A headless collector renders nothing; the rigs would only cost CPU.
            if (Application.isBatchMode) return;
            if (!CaptainCameras || !(captain || CamerasForEveryone)) return;

            var visual = agent.GetComponent<AgentVisual>();
            var rig = agent.gameObject.AddComponent<AgentCameraRig>();
            rig.Build(agent, visual != null ? visual.Animator : null);
            if (Cameras != null) Cameras.Register(rig);
        }

        private void OnAgentDamaged(NpcAgent victim, GameObject attacker)
        {
            if (_firstContactRecorded || attacker == null) return;

            var shooter = attacker.GetComponent<NpcAgent>();
            if (shooter == null) return;

            _firstContactRecorded = true;
            Metrics.RecordFirstContact(shooter.Team);
        }

        private void OnAgentDied(NpcAgent victim, GameObject killer)
        {
            var shooter = killer != null ? killer.GetComponent<NpcAgent>() : null;
            Metrics.RecordKill(shooter != null ? shooter.Team : -1, victim.Team);

            // The body used to be switched off on the spot, which meant the death
            // clips never played and a helmet cam on the victim cut to black. Now it
            // stays and falls, but GoLimp takes it out of the physics -- the rules
            // are the same ones the 19-21 calibration was measured under.
            victim.GoLimp();
            AgentKilled?.Invoke(victim, shooter);
        }

        private IEnumerator RunMatch()
        {
            IsRunning = true;

            for (_round = 0; _round < Rounds; _round++)
            {
                yield return RunRound(_round);
                yield return new WaitForSeconds(IntermissionSeconds);
            }

            IsRunning = false;
            Finish();
        }

        private IEnumerator RunRound(int round)
        {
            // Odd rounds play the mirror image, so spawn advantage cancels out.
            bool swapped = (round & 1) == 1;
            PlaceAgents(swapped);
            _firstContactRecorded = false;
            _roundCenterSeconds[0] = _roundCenterSeconds[1] = 0f;
            RoundStarted?.Invoke(round);

            float deadline = Time.time + RoundTimeLimit;
            if (FixedStep)
                Debug.Log($"[Round] {round} start aliveA={CountAlive(0)} aliveB={CountAlive(1)}");

            while (Time.time < deadline)
            {
                int aliveA = CountAlive(0);
                int aliveB = CountAlive(1);

                PublishScoreline(deadline);
                AccumulateCenterControl(Time.deltaTime);

                if (aliveA == 0 || aliveB == 0)
                {
                    // Both sides wiped in the same frame (a trade) is a draw. It used
                    // to be neither a win nor a draw, so such rounds silently fell
                    // out of the CSV totals; the 19-21 calibration had none.
                    int winner = aliveA == aliveB ? -1 : aliveA > 0 ? 0 : 1;
                    if (winner >= 0) Metrics.RecordRoundWin(winner);
                    else Metrics.RoundsDrawn++;
                    if (FixedStep)
                        Debug.Log($"[Round] {round} elimination winner={winner} aliveA={aliveA} aliveB={aliveB} " +
                                  $"t={RoundTimeLimit - (deadline - Time.time):F1}");
                    RoundEnded?.Invoke(round, winner);
                    yield break;
                }

                yield return null;
            }

            int timeoutWinner = ResolveTimeout();
            if (FixedStep) Debug.Log($"[Round] {round} timeout winner={timeoutWinner}");
            RoundEnded?.Invoke(round, timeoutWinner);
        }

        /// <summary>
        /// Who held the middle decides a stalemate; health only breaks a tie in
        /// that, and a genuine deadlock is recorded as a draw rather than quietly
        /// dropped.
        ///
        /// Calibration has never actually reached this path -- every round so far
        /// ended in an elimination well inside the time limit. It exists so that a
        /// future arena or a more cautious brain cannot turn "hide until the clock
        /// runs out" into a viable strategy, and it is a rules change applied
        /// identically to both teams rather than a tweak to the baseline's
        /// weights, which would have meant handicapping the control group.
        /// </summary>
        /// <returns>The winning team, or -1 for a draw.</returns>
        private int ResolveTimeout()
        {
            float centerA = _roundCenterSeconds[0];
            float centerB = _roundCenterSeconds[1];

            if (Mathf.Abs(centerA - centerB) > 0.5f)
            {
                int winner = centerA > centerB ? 0 : 1;
                Metrics.RecordRoundWin(winner);
                return winner;
            }

            float healthA = TeamHealth(0);
            float healthB = TeamHealth(1);

            if (!Mathf.Approximately(healthA, healthB))
            {
                int winner = healthA > healthB ? 0 : 1;
                Metrics.RecordRoundWin(winner);
                return winner;
            }

            Metrics.RoundsDrawn++;
            return -1;
        }

        private void AccumulateCenterControl(float dt)
        {
            var layout = ArenaLayout.Current;
            if (layout == null) return;

            for (int i = 0; i < _agents.Count; i++)
            {
                var agent = _agents[i];
                if (!agent.IsAlive) continue;
                if (!layout.IsInCenterZone(agent.transform.position)) continue;
                _roundCenterSeconds[agent.Team] += dt;
            }
        }

        private void PlaceAgents(bool swapped)
        {
            NoiseEvents.Clear();
            var layout = ArenaLayout.Current;

            for (int i = 0; i < _agents.Count; i++)
            {
                var agent = _agents[i];
                int slotTeam = swapped ? 1 - agent.Team : agent.Team;
                var slots = layout.TeamSpawns[slotTeam];
                int index = i % AgentsPerTeam;

                var position = slots[Mathf.Min(index, slots.Length - 1)];
                var facing = Quaternion.LookRotation(
                    Vector3.ProjectOnPlane(layout.CenterPosition - position, Vector3.up).normalized);

                agent.ResetForRound(position, facing);

                var visual = agent.GetComponent<AgentVisual>();
                if (visual != null) visual.ResetVisual();
            }
        }

        private void PublishScoreline(float deadline)
        {
            float remaining = Mathf.Clamp01((deadline - Time.time) / Mathf.Max(RoundTimeLimit, 1f));
            float scoreA = Metrics.Teams[0].RoundsWon / (float)Mathf.Max(Rounds, 1);
            float scoreB = Metrics.Teams[1].RoundsWon / (float)Mathf.Max(Rounds, 1);

            for (int i = 0; i < _agents.Count; i++)
            {
                var agent = _agents[i];
                agent.TeamScore01 = agent.Team == 0 ? scoreA : scoreB;
                agent.EnemyScore01 = agent.Team == 0 ? scoreB : scoreA;
                agent.MatchTimeRemaining01 = remaining;
            }
        }

        private int CountAlive(int team)
        {
            int count = 0;
            for (int i = 0; i < _agents.Count; i++)
                if (_agents[i].Team == team && _agents[i].IsAlive) count++;
            return count;
        }

        private float TeamHealth(int team)
        {
            float sum = 0f;
            for (int i = 0; i < _agents.Count; i++)
                if (_agents[i].Team == team && _agents[i].IsAlive) sum += _agents[i].Health.Health01;
            return sum;
        }

        private void Finish()
        {
            if (Recorder != null) Recorder.End();

            string csv = Metrics.ToCsv();
            Debug.Log($"[MatchDirector] {Format} finished after {Rounds} rounds.\n{csv}");

            if (WriteCsvOnFinish)
            {
                string directory = System.IO.Path.Combine(Application.dataPath, "../MatchLogs");
                System.IO.Directory.CreateDirectory(directory);

                // Collection runs explore, so their numbers are not the baseline's;
                // the tag keeps them from ever being read as an evaluation result.
                // Headless runs carry their seed: parallel instances finishing in the
                // same second would otherwise overwrite each other's file.
                string tag = CollectDataset ? "_collect" : _evaluate ? "_eval" : "";
                // Octagon keeps the historical names; any other arena is named in the file.
                var arena = ArenaLayout.Current;
                if (arena != null && arena.Name != "Octagon") tag += "_" + arena.Name;

                // A rules ablation must never be read as a result under the real rules.
                if (MovingSpreadOverride >= 0f)
                    tag += "_spread" + MovingSpreadOverride.ToString("0.##", CultureInfo.InvariantCulture);
                string seed = FixedStep ? $"_seed{Seed}" : "";
                string file = System.IO.Path.Combine(directory,
                    $"{Format}_{TeamABrain}_vs_{TeamBBrain}{tag}_{System.DateTime.Now:yyyyMMdd-HHmmss}{seed}.csv");

                System.IO.File.WriteAllText(file, csv);
                Debug.Log($"[MatchDirector] Wrote {file}");
            }

            if (_quitWhenDone) Application.Quit();
        }
    }
}
