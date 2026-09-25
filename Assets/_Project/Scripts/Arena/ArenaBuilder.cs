using UnityEngine;

namespace JevNpcBrain.Arena
{
    /// <summary>
    /// Builds the "Octagon" arena procedurally and publishes its
    /// <see cref="ArenaLayout"/>.
    ///
    /// Procedural rather than hand-placed for one reason that matters to the
    /// experiment: the cover layout is **point-symmetric by construction**. If the
    /// two spawns do not face identical geometry, the winner of a match is the map,
    /// not the brain, and every number we publish afterwards is noise.
    ///
    /// Phase 5 adds a second, deliberately asymmetric arena here to test whether
    /// the baseline's hand-tuned weights survive a map they were not tuned on.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class ArenaBuilder : MonoBehaviour
    {
        [Header("Shape")]
        [Tooltip("Half-width. 20 m gives the 40 m arena the observation window is sized for.")]
        public float Radius = 20f;

        public float WallHeight = 4f;
        public float CenterZoneRadius = 4f;

        [Header("Cover rings")]
        public float FullCoverRing = 8.5f;
        public float HalfCoverRing = 14f;

        [Header("Squad")]
        [Tooltip("Spawn slots per team. 1 for the captain duel, 3 for the squad match.")]
        [Range(1, 5)] public int SpawnsPerTeam = 3;

        public float SpawnSpread = 3.5f;

        [Header("Debug")]
        public bool DrawOccupancyGizmos = false;

        private Transform _geometry;
        private Material _floorMaterial;
        private Material _coverMaterial;
        private Material _centerMaterial;

        public ArenaLayout Layout { get; private set; }

        private void Awake() => Build();

        [ContextMenu("Build Arena")]
        public void Build()
        {
            ClearGeometry();
            CreateMaterials();

            Layout = new ArenaLayout("Octagon", Radius, transform.position, CenterZoneRadius);

            _geometry = new GameObject("Geometry").transform;
            _geometry.SetParent(transform, false);

            BuildFloor();
            BuildPerimeter();
            BuildCovers();
            BuildCenterZone();
            BuildSpawns();

            ArenaLayout.Current = Layout;
        }

        private void ClearGeometry()
        {
            var existing = transform.Find("Geometry");
            if (existing == null) return;

            if (Application.isPlaying) Destroy(existing.gameObject);
            else DestroyImmediate(existing.gameObject);
        }

        /// <summary>URP first, then built-in. Greybox only -- the look pass comes later.</summary>
        private void CreateMaterials()
        {
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");

            _floorMaterial = new Material(shader) { name = "ArenaFloor" };
            _floorMaterial.color = new Color(0.22f, 0.23f, 0.26f);

            _coverMaterial = new Material(shader) { name = "ArenaCover" };
            _coverMaterial.color = new Color(0.45f, 0.46f, 0.50f);

            _centerMaterial = new Material(shader) { name = "ArenaCenter" };
            _centerMaterial.color = new Color(0.85f, 0.62f, 0.25f);
        }

        private void BuildFloor()
        {
            var floor = Block("Floor", Vector3.down * 0.5f, Quaternion.identity,
                new Vector3(Radius * 2f + 4f, 1f, Radius * 2f + 4f), _floorMaterial);
            floor.isStatic = true;
        }

        /// <summary>Eight wall segments, one per octagon edge.</summary>
        private void BuildPerimeter()
        {
            const int sides = 8;
            float apothem = Radius * Mathf.Cos(Mathf.PI / sides);
            float edge = 2f * Radius * Mathf.Sin(Mathf.PI / sides);

            for (int i = 0; i < sides; i++)
            {
                float angle = (i + 0.5f) * Mathf.PI * 2f / sides;
                var outward = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));

                Block("Wall_" + i,
                    outward * (apothem + 0.5f) + Vector3.up * (WallHeight * 0.5f),
                    Quaternion.LookRotation(outward),
                    new Vector3(edge + 1f, WallHeight, 1f),
                    _coverMaterial);
            }
        }

        /// <summary>
        /// Four full-height and eight crouch-height blocks, both rings placed at
        /// evenly spaced angles -- which makes the whole layout invariant under a
        /// 180 degree rotation, the symmetry the fairness contract needs.
        /// </summary>
        private void BuildCovers()
        {
            for (int i = 0; i < 4; i++)
            {
                float angle = (45f + i * 90f) * Mathf.Deg2Rad;
                PlaceCover("FullCover_" + i, angle, FullCoverRing,
                    new Vector3(3f, 2.6f, 1.4f), 1f);
            }

            for (int i = 0; i < 8; i++)
            {
                float angle = (i * 45f) * Mathf.Deg2Rad;
                PlaceCover("HalfCover_" + i, angle, HalfCoverRing,
                    new Vector3(3.4f, 1.1f, 1.2f), 0.5f);
            }
        }

        private void PlaceCover(string name, float angle, float ring, Vector3 size, float height)
        {
            var radial = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
            var center = transform.position + radial * ring;

            Block(name, center - transform.position + Vector3.up * (size.y * 0.5f),
                Quaternion.LookRotation(radial), size, _coverMaterial);

            Stamp(center, size, Quaternion.LookRotation(radial), height);

            // Two standing spots, one on each face. Each shelters from the side the
            // block is between you and -- hence the opposing normals.
            float standoff = size.z * 0.5f + 0.9f;
            Layout.CoverAnchors.Add(new CoverAnchor(center + radial * standoff, radial, height));
            Layout.CoverAnchors.Add(new CoverAnchor(center - radial * standoff, -radial, height));
        }

        private void BuildCenterZone()
        {
            var zone = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            zone.name = "CenterZone";
            zone.transform.SetParent(_geometry, false);
            zone.transform.localPosition = Vector3.up * 0.02f;
            zone.transform.localScale = new Vector3(CenterZoneRadius * 2f, 0.02f, CenterZoneRadius * 2f);
            zone.GetComponent<MeshRenderer>().sharedMaterial = _centerMaterial;

            // Flat marker, not an obstacle -- agents must be able to walk onto it.
            var collider = zone.GetComponent<Collider>();
            if (Application.isPlaying) Destroy(collider);
            else DestroyImmediate(collider);
        }

        /// <summary>
        /// Spawns sit on the north and south apothem, mirrored through the origin.
        /// MatchDirector swaps which team gets which set between rounds so any
        /// residual advantage cancels out.
        /// </summary>
        private void BuildSpawns()
        {
            float inset = Radius - 3f;

            for (int team = 0; team < 2; team++)
            {
                var slots = new Vector3[SpawnsPerTeam];
                float sign = team == 0 ? -1f : 1f;

                for (int i = 0; i < SpawnsPerTeam; i++)
                {
                    float lateral = (i - (SpawnsPerTeam - 1) * 0.5f) * SpawnSpread;
                    slots[i] = transform.position + new Vector3(lateral * sign, 0f, inset * sign);
                }

                Layout.TeamSpawns[team] = slots;
            }
        }

        private GameObject Block(string name, Vector3 localPosition, Quaternion rotation,
            Vector3 size, Material material)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(_geometry, false);
            go.transform.localPosition = localPosition;
            go.transform.localRotation = rotation;
            go.transform.localScale = size;
            go.GetComponent<MeshRenderer>().sharedMaterial = material;
            return go;
        }

        /// <summary>
        /// Rasterises a rotated box into the occupancy grid. Sampling the box in its
        /// own local space keeps rotated covers from bleeding into cells they do not
        /// actually occupy, which would show up later as an agent hugging thin air.
        /// </summary>
        private void Stamp(Vector3 center, Vector3 size, Quaternion rotation, float height)
        {
            var inverse = Quaternion.Inverse(rotation);
            var half = new Vector2(size.x * 0.5f, size.z * 0.5f);
            float reach = Mathf.Max(size.x, size.z) * 0.5f + ArenaLayout.CellSize;

            Layout.TryWorldToCell(center - Vector3.one * reach, out int minX, out int minY);
            Layout.TryWorldToCell(center + Vector3.one * reach, out int maxX, out int maxY);

            minX = Mathf.Max(minX, 0);
            minY = Mathf.Max(minY, 0);
            maxX = Mathf.Min(maxX, Layout.Resolution - 1);
            maxY = Mathf.Min(maxY, Layout.Resolution - 1);

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    var local = inverse * (Layout.CellToWorld(x, y) - center);
                    if (Mathf.Abs(local.x) <= half.x && Mathf.Abs(local.z) <= half.y)
                        Layout.SetHeight(x, y, height);
                }
            }
        }

        private void OnDrawGizmosSelected()
        {
            if (Layout == null) return;

            if (DrawOccupancyGizmos)
            {
                for (int y = 0; y < Layout.Resolution; y++)
                {
                    for (int x = 0; x < Layout.Resolution; x++)
                    {
                        float h = Layout.SampleHeight(x, y);
                        if (h <= 0f) continue;

                        Gizmos.color = h >= 1f
                            ? new Color(1f, 0.3f, 0.3f, 0.35f)
                            : new Color(1f, 0.85f, 0.3f, 0.35f);
                        Gizmos.DrawCube(Layout.CellToWorld(x, y) + Vector3.up * 0.1f,
                            new Vector3(ArenaLayout.CellSize, 0.1f, ArenaLayout.CellSize));
                    }
                }
            }

            Gizmos.color = Color.cyan;
            foreach (var anchor in Layout.CoverAnchors)
            {
                Gizmos.DrawWireSphere(anchor.Position + Vector3.up, 0.4f);
                Gizmos.DrawRay(anchor.Position + Vector3.up, anchor.Normal * 1.5f);
            }

            for (int team = 0; team < 2; team++)
            {
                var slots = Layout.TeamSpawns[team];
                if (slots == null) continue;

                Gizmos.color = team == 0 ? Color.blue : Color.red;
                foreach (var slot in slots) Gizmos.DrawWireCube(slot + Vector3.up, Vector3.one);
            }
        }
    }
}
