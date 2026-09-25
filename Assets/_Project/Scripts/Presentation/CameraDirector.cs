using System.Collections.Generic;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace JevNpcBrain.Presentation
{
    using Core;
    using Match;

    public enum ShotKind { Tactical, Duel, Wide, Tower, Orbit, Helmet, Shoulder }

    /// <summary>
    /// Owns every shot in the match and decides what is on screen.
    ///
    /// Two layouts. Single shows one shot with the tactical view inset in a
    /// corner -- a viewer watching down one captain's sights has otherwise lost
    /// the picture that makes the comparison legible. Versus splits the screen
    /// between the two captains' helmet cams, which is the premise of the whole
    /// project in one frame: same arena, same rifle, two different brains.
    ///
    /// Keys belong to kinds of shot, not to registration order, so a key always
    /// means the same camera whatever spawned first:
    ///   1 tactical  2 duel  3 wide  4 towers (press again to step)  5 orbit
    ///   6 / 7  helmet cam, team A / B captain
    ///   8 / 9  over the shoulder, team A / B captain
    ///   Tab next (Shift+Tab back)  V versus  P inset  O auto-director
    ///
    /// Off-air cameras have their Camera component disabled, never their
    /// GameObject, so every rig keeps tracking and a cut never lands on a stale
    /// frame.
    /// </summary>
    [DefaultExecutionOrder(200)]
    public sealed class CameraDirector : MonoBehaviour
    {
        public sealed class Shot
        {
            public string Label;
            public Camera Camera;
            public ShotKind Kind;
            public NpcAgent Agent;

            /// <summary>Set for arena cameras; the towers report who they are following.</summary>
            public BroadcastCamera Broadcast;
        }

        private enum Situation { Quiet, Opening, Contact, Aftermath }

        [Header("Spectator")]
        public Camera SpectatorCamera;

        [Header("Picture in picture")]
        public bool PictureInPicture = true;
        public Rect InsetRect = new Rect(0.755f, 0.03f, 0.23f, 0.23f);

        [Tooltip("Tactical inset in the versus split: top centre, clear of both rifles.")]
        public Rect VersusInsetRect = new Rect(0.4f, 0.66f, 0.2f, 0.26f);

        [Header("Versus")]
        [Tooltip("Split screen: team A captain's helmet cam left, team B's right.")]
        public bool Versus;

        [Header("Auto director")]
        public bool AutoDirect;

        [Tooltip("Real seconds a shot is held before the director may cut away.")]
        public float MinShotSeconds = 3.5f;

        [Tooltip("Real seconds before a shot is cut for variety, even if nothing changed.")]
        public float MaxShotSeconds = 8f;

        [Tooltip("On a kill, cut straight to the killer's helmet cam and watch the body drop.")]
        public bool KillCams = true;

        public float OpeningSeconds = 3f;
        public float AftermathSeconds = 2.5f;

        [Header("Wiring")]
        public MatchDirector Match;

        private static readonly Rect FullScreen = new Rect(0f, 0f, 1f, 1f);
        private static readonly Rect LeftHalf = new Rect(0f, 0f, 0.5f, 1f);
        private static readonly Rect RightHalf = new Rect(0.5f, 0f, 0.5f, 1f);

#if ENABLE_INPUT_SYSTEM
        private static readonly Key[] DigitKeys =
        {
            Key.Digit1, Key.Digit2, Key.Digit3, Key.Digit4, Key.Digit5,
            Key.Digit6, Key.Digit7, Key.Digit8, Key.Digit9
        };

        private static readonly Key[] NumpadKeys =
        {
            Key.Numpad1, Key.Numpad2, Key.Numpad3, Key.Numpad4, Key.Numpad5,
            Key.Numpad6, Key.Numpad7, Key.Numpad8, Key.Numpad9
        };
#endif

        private readonly List<Shot> _shots = new List<Shot>(16);
        private readonly List<Shot> _candidates = new List<Shot>(16);
        private readonly List<float> _weights = new List<float>(16);

        private Shot _live;
        private float _cutAt = float.NegativeInfinity;
        private Situation _shotSituation;
        private bool _forceCut;
        private float _roundStartedAt = float.NegativeInfinity;
        private float _killAt = float.NegativeInfinity;
        private NpcAgent _lastKiller;
        private bool _versusOnAir;
        private bool _appliedVersus;
        private bool _appliedInset;

        public int ShotCount => _shots.Count;
        public bool VersusOnAir => _versusOnAir;

        public string CurrentLabel
        {
            get
            {
                if (_versusOnAir && TryGetVersusPair(out var left, out var right))
                    return "[V] " + left.Agent.DisplayName + "  |  " + right.Agent.DisplayName;

                if (_live == null) return "none";

                int key = KeyOf(_live);
                return (key > 0 ? "[" + key + "] " : "") + _live.Label;
            }
        }

        private void Awake()
        {
            if (SpectatorCamera == null) SpectatorCamera = Camera.main;
            if (SpectatorCamera != null) Register(SpectatorCamera, ShotKind.Tactical, "Tactical");
        }

        private void Start()
        {
            if (Match == null) Match = FindAnyObjectByType<MatchDirector>();

            if (Match != null)
            {
                Match.RoundStarted += OnRoundStarted;
                Match.AgentKilled += OnAgentKilled;
            }

            Apply();
        }

        private void OnDestroy()
        {
            if (Match == null) return;
            Match.RoundStarted -= OnRoundStarted;
            Match.AgentKilled -= OnAgentKilled;
        }

        public void Register(Camera camera, ShotKind kind, string label, NpcAgent agent = null)
        {
            if (camera == null) return;

            var shot = new Shot
            {
                Label = label,
                Camera = camera,
                Kind = kind,
                Agent = agent,
                Broadcast = camera.GetComponent<BroadcastCamera>()
            };

            _shots.Add(shot);
            if (_live == null) _live = shot;

            // Registration keeps arriving after Start -- agents spawn a frame late.
            // Re-assert the layout so a new camera never goes on air by accident.
            Apply();
        }

        public void Register(AgentCameraRig rig)
        {
            if (rig == null || rig.Agent == null) return;

            string who = rig.Agent.DisplayName;
            Register(rig.Helmet, ShotKind.Helmet, who + " / Helmet", rig.Agent);
            Register(rig.OverShoulder, ShotKind.Shoulder, who + " / Shoulder", rig.Agent);
        }

        /// <summary>
        /// The number key a shot answers to, worked out on demand rather than
        /// stored, so it can never go stale if an agent's team or captaincy is
        /// assigned after its cameras exist.
        /// </summary>
        public static int KeyOf(Shot shot)
        {
            switch (shot.Kind)
            {
                case ShotKind.Tactical: return 1;
                case ShotKind.Duel: return 2;
                case ShotKind.Wide: return 3;
                case ShotKind.Tower: return 4;
                case ShotKind.Orbit: return 5;
                case ShotKind.Helmet: return CaptainKey(shot.Agent, 6);
                case ShotKind.Shoulder: return CaptainKey(shot.Agent, 8);
                default: return 0;
            }
        }

        private static int CaptainKey(NpcAgent agent, int first)
            => agent != null && agent.IsCaptain && agent.Team >= 0 && agent.Team <= 1
                ? first + agent.Team
                : 0;

        private void Update()
        {
            ReadInput();

            // Ticking Versus or the inset in the Inspector mid-match works like the keys.
            if (Versus != _appliedVersus || PictureInPicture != _appliedInset) Apply();

            if (AutoDirect && !Versus) TickAutoDirector();
        }

        private void ReadInput()
        {
#if ENABLE_INPUT_SYSTEM
            var keyboard = Keyboard.current;
            if (keyboard == null) return;

            if (keyboard.tabKey.wasPressedThisFrame) Cycle(keyboard.shiftKey.isPressed ? -1 : 1);
            if (keyboard.vKey.wasPressedThisFrame) ToggleVersus();
            if (keyboard.pKey.wasPressedThisFrame) TogglePictureInPicture();
            if (keyboard.oKey.wasPressedThisFrame) ToggleAutoDirect();

            for (int i = 0; i < DigitKeys.Length; i++)
            {
                if (!keyboard[DigitKeys[i]].wasPressedThisFrame &&
                    !keyboard[NumpadKeys[i]].wasPressedThisFrame) continue;

                SelectKey(i + 1);
                return;
            }
#endif
        }

        /// <summary>
        /// Several shots can share a key -- all four towers live on one -- so
        /// pressing it again steps to the next shot with that key.
        /// </summary>
        public void SelectKey(int key)
        {
            AutoDirect = false;
            Versus = false;

            int start = _live != null && KeyOf(_live) == key ? _shots.IndexOf(_live) + 1 : 0;

            for (int n = 0; n < _shots.Count; n++)
            {
                var shot = _shots[(start + n) % _shots.Count];
                if (shot.Camera == null || KeyOf(shot) != key) continue;

                Cut(shot, _shotSituation);
                return;
            }

            Apply();
        }

        public void Cycle(int step)
        {
            if (_shots.Count == 0) return;

            AutoDirect = false;
            Versus = false;

            int index = Mathf.Max(_shots.IndexOf(_live), 0);
            for (int n = 0; n < _shots.Count; n++)
            {
                index = (index + step + _shots.Count) % _shots.Count;
                if (_shots[index].Camera == null) continue;

                Cut(_shots[index], _shotSituation);
                return;
            }
        }

        public void ToggleVersus()
        {
            Versus = !Versus;
            if (Versus) AutoDirect = false;
            Apply();
        }

        public void TogglePictureInPicture()
        {
            PictureInPicture = !PictureInPicture;
            Apply();
        }

        public void ToggleAutoDirect()
        {
            AutoDirect = !AutoDirect;

            if (AutoDirect)
            {
                Versus = false;
                _forceCut = true;
            }

            Apply();
        }

        private void Cut(Shot shot, Situation situation)
        {
            _live = shot;
            _cutAt = Time.unscaledTime;
            _shotSituation = situation;
            Apply();
        }

        private void Apply()
        {
            _appliedVersus = Versus;
            _appliedInset = PictureInPicture;

            for (int i = 0; i < _shots.Count; i++)
                if (_shots[i].Camera != null) _shots[i].Camera.enabled = false;

            _versusOnAir = false;

            if (Versus && TryGetVersusPair(out var left, out var right))
            {
                Show(left.Camera, LeftHalf, 0f);
                Show(right.Camera, RightHalf, 0f);
                if (PictureInPicture) ShowInset(VersusInsetRect);
                _versusOnAir = true;
                return;
            }

            if ((_live == null || _live.Camera == null) && _shots.Count > 0) _live = _shots[0];
            if (_live == null || _live.Camera == null) return;

            Show(_live.Camera, FullScreen, 0f);

            // The tactical view stays in the corner whenever it is not the main shot.
            if (PictureInPicture && _live.Camera != SpectatorCamera) ShowInset(InsetRect);
        }

        private static void Show(Camera camera, Rect rect, float depth)
        {
            camera.rect = rect;
            camera.depth = depth;
            camera.enabled = true;
        }

        /// <summary>Higher depth so it draws last, over whatever is on air.</summary>
        private void ShowInset(Rect rect)
        {
            if (SpectatorCamera != null) Show(SpectatorCamera, rect, 10f);
        }

        private bool TryGetVersusPair(out Shot left, out Shot right)
        {
            left = right = null;

            for (int i = 0; i < _shots.Count; i++)
            {
                var shot = _shots[i];
                if (shot.Kind != ShotKind.Helmet || shot.Camera == null) continue;
                if (shot.Agent == null || !shot.Agent.IsCaptain) continue;

                if (shot.Agent.Team == 0) left = shot;
                else if (shot.Agent.Team == 1) right = shot;
            }

            return left != null && right != null;
        }

        /// <summary>A hard seam down the middle, so the split reads as deliberate.</summary>
        private void OnGUI()
        {
            if (!_versusOnAir) return;

            var previous = GUI.color;
            GUI.color = Color.black;
            GUI.DrawTexture(new Rect(Screen.width * 0.5f - 2f, 0f, 4f, Screen.height), Texture2D.whiteTexture);
            GUI.color = previous;
        }

        private void OnRoundStarted(int round) => _roundStartedAt = Time.unscaledTime;

        /// <summary>
        /// A kill is worth breaking the minimum hold for -- it is the payoff of
        /// the round. From the killer's helmet the viewer sees the body drop.
        /// </summary>
        private void OnAgentKilled(NpcAgent victim, NpcAgent killer)
        {
            _killAt = Time.unscaledTime;
            _lastKiller = killer;

            if (!AutoDirect || Versus) return;

            var killcam = KillCams ? Find(ShotKind.Helmet, killer) : null;
            if (killcam != null) Cut(killcam, Situation.Aftermath);
            else _forceCut = true;
        }

        private Shot Find(ShotKind kind, NpcAgent agent)
        {
            if (agent == null) return null;

            for (int i = 0; i < _shots.Count; i++)
                if (_shots[i].Kind == kind && _shots[i].Agent == agent && _shots[i].Camera != null)
                    return _shots[i];

            return null;
        }

        /// <summary>
        /// Holds a shot long enough to be watchable, then cuts when the situation
        /// changes -- or after a while anyway, so a long stand-off does not sit
        /// on one angle. All timing is in real seconds: how long a viewer has
        /// been looking at something does not speed up with the simulation.
        /// </summary>
        private void TickAutoDirector()
        {
            float now = Time.unscaledTime;
            var situation = Assess(now);
            float held = now - _cutAt;

            bool due = _forceCut ||
                       (held >= MinShotSeconds && (situation != _shotSituation || held >= MaxShotSeconds));
            if (!due) return;

            _forceCut = false;

            var next = Pick(situation);
            if (next != null && next != _live)
            {
                Cut(next, situation);
                return;
            }

            // Nothing better to cut to: keep the shot and look again later.
            _cutAt = now;
            _shotSituation = situation;
        }

        private Situation Assess(float now)
        {
            if (now - _killAt < AftermathSeconds) return Situation.Aftermath;
            if (now - _roundStartedAt < OpeningSeconds) return Situation.Opening;

            var all = NpcAgent.All;
            for (int i = 0; i < all.Count; i++)
                if (Sees(all[i])) return Situation.Contact;

            return Situation.Quiet;
        }

        private Shot Pick(Situation situation)
        {
            _candidates.Clear();
            _weights.Clear();

            var bestTower = BestTower();
            float total = 0f;

            for (int i = 0; i < _shots.Count; i++)
            {
                var shot = _shots[i];
                if (shot == _live || shot.Camera == null) continue;
                if (shot.Kind == ShotKind.Tower && shot != bestTower) continue;

                float weight = Weight(shot, situation);
                if (weight <= 0f) continue;

                _candidates.Add(shot);
                _weights.Add(weight);
                total += weight;
            }

            if (_candidates.Count == 0) return null;

            float roll = Random.value * total;
            for (int i = 0; i < _candidates.Count; i++)
            {
                roll -= _weights[i];
                if (roll <= 0f) return _candidates[i];
            }

            return _candidates[_candidates.Count - 1];
        }

        /// <summary>What each kind of shot is worth in each situation.</summary>
        private float Weight(Shot shot, Situation situation)
        {
            var agent = shot.Agent;
            bool alive = agent != null && agent.IsAlive;
            bool sees = Sees(agent);

            switch (situation)
            {
                case Situation.Opening:
                    if (shot.Kind == ShotKind.Orbit) return 3f;
                    if (shot.Kind == ShotKind.Wide) return 2f;
                    return shot.Kind == ShotKind.Tactical ? 1f : 0f;

                case Situation.Aftermath:
                    if (shot.Kind == ShotKind.Helmet && agent == _lastKiller) return 3f;
                    if (shot.Kind == ShotKind.Shoulder && agent == _lastKiller) return 2f;
                    if (shot.Kind == ShotKind.Duel) return 2f;
                    return shot.Kind == ShotKind.Wide ? 1f : 0f;

                case Situation.Contact:
                    switch (shot.Kind)
                    {
                        case ShotKind.Duel: return 3f;
                        case ShotKind.Helmet: return sees ? 2f : 0f;
                        case ShotKind.Shoulder: return sees ? 1.5f : 0f;
                        case ShotKind.Tower: return 1f;
                        case ShotKind.Wide: return 1f;
                        default: return 0f;
                    }

                default:
                    switch (shot.Kind)
                    {
                        case ShotKind.Tactical: return 2f;
                        case ShotKind.Wide: return 1.5f;
                        case ShotKind.Tower: return 1.5f;
                        case ShotKind.Orbit: return 1f;
                        case ShotKind.Shoulder: return alive && agent.IsCaptain ? 0.75f : 0f;
                        case ShotKind.Duel: return 0.5f;
                        default: return 0f;
                    }
            }
        }

        private static bool Sees(NpcAgent agent)
            => agent != null && agent.IsAlive && agent.Observation.Self.HasLineOfSight > 0.5f;

        /// <summary>
        /// The one tower worth cutting to: closest to its fighter, among those that
        /// can actually see them. A long lens on a body hidden behind cover is the
        /// one shot that looks broken, so an occluded tower is never picked.
        /// </summary>
        private Shot BestTower()
        {
            Shot best = null;
            float bestDistance = float.MaxValue;

            for (int i = 0; i < _shots.Count; i++)
            {
                var shot = _shots[i];
                if (shot.Kind != ShotKind.Tower || shot.Camera == null) continue;

                var rig = shot.Broadcast;
                if (rig == null || rig.Subject == null || !rig.SubjectInView) continue;

                float distance = (shot.Camera.transform.position - rig.Subject.transform.position).sqrMagnitude;
                if (distance >= bestDistance) continue;

                bestDistance = distance;
                best = shot;
            }

            return best;
        }
    }
}
