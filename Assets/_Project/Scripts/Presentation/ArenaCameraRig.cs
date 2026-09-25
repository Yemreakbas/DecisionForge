using UnityEngine;

namespace JevNpcBrain.Presentation
{
    using Arena;

    /// <summary>
    /// Mounts the broadcast cameras around the arena and hands them to the
    /// director. Built from the ArenaLayout rather than placed by hand, for the
    /// same reason the arena itself is: phase 5 brings a second arena, and the
    /// cameras should follow the geometry there without anyone re-rigging them.
    ///
    ///   Wide   -- high over the east sideline. The establishing shot, both
    ///             spawns left and right, like a pitch seen from the stands.
    ///   Towers -- one above each diagonal corner, a long lens on the nearest fighter.
    ///   Duel   -- the side-on two-shot of the captains.
    ///   Orbit  -- a slow circle, for round openers.
    /// </summary>
    public sealed class ArenaCameraRig : MonoBehaviour
    {
        public CameraDirector Cameras;

        [Header("Wide")]
        [Tooltip("Metres outside the arena radius, on the east side.")]
        public float WideOutside = 7f;
        public float WideHeight = 13f;

        [Header("Towers")]
        public float TowerHeight = 8f;

        [Tooltip("Metres in from the arena's corners.")]
        public float TowerInset = 2f;

        [Header("Orbit")]
        public float OrbitHeight = 9f;
        public float OrbitInset = 3f;

        private static readonly string[] TowerNames = { "Tower NE", "Tower NW", "Tower SW", "Tower SE" };

        /// <summary>Start, not Awake: ArenaBuilder publishes the layout in its own Awake.</summary>
        private void Start()
        {
            if (Cameras == null) Cameras = FindAnyObjectByType<CameraDirector>();

            var layout = ArenaLayout.Current;
            if (layout == null)
            {
                Debug.LogError("[ArenaCameraRig] No ArenaLayout. Is ArenaBuilder in the scene?");
                return;
            }

            Build(layout);
        }

        private void Build(ArenaLayout layout)
        {
            var center = layout.CenterPosition;

            var wide = Make("Wide", BroadcastCamera.Rig.Track,
                center + new Vector3(layout.Radius + WideOutside, WideHeight, 0f));
            wide.MinFieldOfView = 26f;
            Register(wide, ShotKind.Wide, "Wide");

            for (int i = 0; i < TowerNames.Length; i++)
            {
                float angle = (45f + i * 90f) * Mathf.Deg2Rad;
                var corner = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));

                var tower = Make(TowerNames[i], BroadcastCamera.Rig.Track,
                    center + corner * (layout.Radius - TowerInset) + Vector3.up * TowerHeight);
                tower.FollowNearest = true;
                tower.MinFieldOfView = 12f;
                tower.MaxFieldOfView = 40f;
                Register(tower, ShotKind.Tower, TowerNames[i]);
            }

            var duel = Make("Duel", BroadcastCamera.Rig.Duel, center + new Vector3(10f, 3f, 0f));
            Register(duel, ShotKind.Duel, "Duel");

            float orbitRadius = layout.Radius - OrbitInset;
            var orbit = Make("Orbit", BroadcastCamera.Rig.Orbit,
                center + new Vector3(0f, OrbitHeight, -orbitRadius));
            orbit.OrbitRadius = orbitRadius;
            orbit.OrbitHeight = OrbitHeight;
            Register(orbit, ShotKind.Orbit, "Orbit");
        }

        private BroadcastCamera Make(string label, BroadcastCamera.Rig rig, Vector3 position)
        {
            var go = new GameObject("Cam_" + label);
            go.transform.SetParent(transform, false);
            go.transform.position = position;

            var camera = go.AddComponent<Camera>();
            camera.nearClipPlane = 0.3f;
            camera.farClipPlane = 300f;

            // Off air until the director cuts to it.
            camera.enabled = false;

            var broadcast = go.AddComponent<BroadcastCamera>();
            broadcast.Mode = rig;
            return broadcast;
        }

        private void Register(BroadcastCamera shot, ShotKind kind, string label)
        {
            if (Cameras == null)
            {
                Debug.LogWarning("[ArenaCameraRig] No CameraDirector in the scene; " + label + " will never go on air.");
                return;
            }

            Cameras.Register(shot.GetComponent<Camera>(), kind, label);
        }
    }
}
