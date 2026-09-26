using System.Collections.Generic;
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
    /// Phase 5 adds "Warehouse", a held-out arena to test whether the baseline's
    /// hand-tuned weights -- and the world model trained only on Octagon --
    /// survive a map neither was built on. It is structurally unlike Octagon
    /// (square, long lane-forming walls, diagonal spawns, crates at the centre)
    /// but still point-symmetric: a map that favoured one spawn would measure the
    /// map, not the brain.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class ArenaBuilder : MonoBehaviour
    {
        /// <summary>Serialized by value in the scene: append only.</summary>
        public enum ArenaKind { Octagon, Warehouse, Divide, Procedural }

        [Header("Layout")]
        [Tooltip("Octagon: where the baseline was tuned and the world model trained. Warehouse and " +
                 "Divide: phase 5 held-out arenas. A player's '-arena <name>' overrides this.")]
        public ArenaKind Kind = ArenaKind.Octagon;

        [Tooltip("Procedural only: which layout. A player's '-layoutseed N' overrides it.")]
        public int LayoutSeed = 1;

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

        private void Awake()
        {
            ApplyCommandLine();
            Build();
        }

        /// <summary>
        /// Read here rather than by MatchDirector: the arena builds in Awake, a
        /// frame before the director parses its own flags in Start.
        /// </summary>
        private void ApplyCommandLine()
        {
            var args = System.Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; i++)
            {
                string flag = args[i].ToLowerInvariant();
                if (flag == "-arena" && System.Enum.TryParse(args[i + 1], true, out ArenaKind kind))
                    Kind = kind;
                else if (flag == "-layoutseed" && int.TryParse(args[i + 1],
                             System.Globalization.NumberStyles.Integer,
                             System.Globalization.CultureInfo.InvariantCulture, out int seed))
                    LayoutSeed = seed;
            }
        }

        [ContextMenu("Build Arena")]
        public void Build()
        {
            ClearGeometry();
            CreateMaterials();

            string layoutName = Kind == ArenaKind.Procedural ? "Proc" + LayoutSeed : Kind.ToString();
            Layout = new ArenaLayout(layoutName, Radius, transform.position, CenterZoneRadius);

            _geometry = new GameObject("Geometry").transform;
            _geometry.SetParent(transform, false);
            _pairs = 0;

            BuildFloor();
            switch (Kind)
            {
                case ArenaKind.Warehouse:
                    BuildSquarePerimeter();
                    BuildWarehouseCovers();
                    break;
                case ArenaKind.Divide:
                    BuildSquarePerimeter();
                    BuildDivideCovers();
                    break;
                case ArenaKind.Procedural:
                    BuildProceduralLayout();   // perimeter, spawns and pieces
                    break;
                default:
                    BuildPerimeter();
                    BuildCovers();
                    break;
            }
            BuildCenterZone();
            if (Kind == ArenaKind.Warehouse) BuildDiagonalSpawns();
            else if (Kind != ArenaKind.Procedural) BuildSpawns();

            ArenaLayout.Current = Layout;
        }

        private int _pairs;

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

        // ---- Warehouse (phase 5, held out) ---------------------------------------

        private void BuildSquarePerimeter()
        {
            for (int i = 0; i < 4; i++)
            {
                var outward = Quaternion.Euler(0f, i * 90f, 0f) * Vector3.forward;
                Block("Wall_" + i,
                    outward * (Radius + 0.5f) + Vector3.up * (WallHeight * 0.5f),
                    Quaternion.LookRotation(outward),
                    new Vector3(Radius * 2f + 2f, WallHeight, 1f),
                    _coverMaterial);
            }
        }

        /// <summary>
        /// Eight full-height and twelve crouch-height pieces, every one placed as a
        /// pair through the origin. The long shelves and partitions cut the floor
        /// into lanes Octagon never had; the columns on each spawn diagonal hide
        /// the centre from the spawns; the crates on the centre's rim give the
        /// contested zone cover it never had either.
        /// </summary>
        private void BuildWarehouseCovers()
        {
            const float full = 2.6f, crouch = 1.1f;

            // Stamp marks a cell only if its *centre* is inside the box, and cell
            // centres sit at -19.375 + 1.25k. A 1.2 m wall centred between two of
            // them marks nothing: solid to physics, invisible to line of sight and
            // to the observation grid. So every thin axis below is centred on a
            // cell centre and every short crate spans a cell boundary. (Stamp
            // itself stays as it is: changing it would change Octagon's
            // observations under the trained world model.)
            PlacePair("Shelf", -6f, 5.625f, 0f, new Vector3(9f, full, 1.4f), 1f, 3);
            PlacePair("Partition", -11.875f, -4f, 90f, new Vector3(7f, full, 1.4f), 1f, 2);
            PlacePair("Pillar", 3.125f, 10.625f, 0f, new Vector3(1.6f, full, 1.6f), 1f, 1);
            PlacePair("Column", 8.125f, 8.125f, 45f, new Vector3(2.2f, full, 2.2f), 1f, 1);

            var crate = new Vector3(2.6f, crouch, 1.2f);
            PlacePair("Crate", -15f, 11.875f, 0f, crate, 0.5f, 1);
            PlacePair("Crate", -1.875f, -5f, 90f, crate, 0.5f, 1);
            PlacePair("Crate", 10f, -14.375f, 0f, crate, 0.5f, 1);
            PlacePair("Crate", -16.875f, 1.25f, 90f, crate, 0.5f, 1);
            PlacePair("Crate", 4.375f, -1.25f, 90f, crate, 0.5f, 1);
            PlacePair("Crate", -11.875f, -11.875f, 45f, crate, 0.5f, 1);
        }

        // ---- Procedural (phase 5c: many arenas for the world model) ---------------

        /// <summary>A piece in whole cells: lower-left cell, width along x, depth along z.</summary>
        private readonly struct Footprint
        {
            public readonly int X, Y, W, D;
            public readonly float Height;

            public Footprint(int x, int y, int w, int d, float height)
            {
                X = x; Y = y; W = w; D = d; Height = height;
            }

            /// <summary>The same piece turned 180 degrees about the arena centre.</summary>
            public Footprint Mirror(int res) => new Footprint(res - X - W, res - Y - D, W, D, Height);
        }

        /// <summary>
        /// A seeded random layout under the rules three hand-made arenas taught:
        /// pieces are whole cells (so Stamp marks exactly what physics blocks),
        /// every piece has its twin through the origin, a cell of floor is kept
        /// around each piece, spawns and the centre stay clear, the two captains
        /// see each other from their spawns (neither brain searches for an enemy
        /// it has never seen), and every spawn can walk to every other. A piece
        /// that would break a rule is not placed. Same seed, same arena.
        /// </summary>
        private void BuildProceduralLayout()
        {
            var rng = new System.Random(LayoutSeed);
            BuildSquarePerimeter();
            if (rng.Next(2) == 1) BuildDiagonalSpawns();
            else BuildSpawns();

            int res = Layout.Resolution;
            var heights = new float[res * res];
            var placed = new List<Footprint>();
            int pairs = rng.Next(6, 12);

            for (int attempt = 0; attempt < 400 && placed.Count < pairs * 2; attempt++)
            {
                var piece = RandomFootprint(rng, res);
                var twin = piece.Mirror(res);
                if (!IsFree(piece, heights, res) || !IsFree(twin, heights, res)) continue;
                if (Touches(piece, twin) || TooCloseToSpawnOrCentre(piece) || TooCloseToSpawnOrCentre(twin)) continue;

                Fill(piece, heights, res, piece.Height);
                Fill(twin, heights, res, twin.Height);
                if (!CaptainsSeeEachOther(heights, res) || !SpawnsConnected(heights, res))
                {
                    Fill(piece, heights, res, 0f);
                    Fill(twin, heights, res, 0f);
                    continue;
                }
                placed.Add(piece);
                placed.Add(twin);
            }

            for (int i = 0; i < placed.Count; i++)
            {
                var f = placed[i];
                bool alongX = f.W >= f.D;
                int length = Mathf.Max(f.W, f.D), thickness = Mathf.Min(f.W, f.D);
                var local = new Vector3(-Radius + ArenaLayout.CellSize * (f.X + f.W * 0.5f), 0f,
                                        -Radius + ArenaLayout.CellSize * (f.Y + f.D * 0.5f));
                // 0.1 m short of whole cells: covers exactly length x thickness cell centres.
                var size = new Vector3(ArenaLayout.CellSize * length - 0.1f, f.Height >= 1f ? 2.6f : 1.1f,
                                       ArenaLayout.CellSize * thickness - 0.1f);
                string kind = f.Height < 1f ? "Crate" : length >= 4 ? "Wall" : length == 2 ? "Block" : "Pillar";
                PlaceBox($"{kind}_{i / 2}_{(i % 2 == 0 ? "A" : "B")}", local, alongX ? 0f : 90f, size, f.Height,
                    length >= 4 ? 3 : 1);
            }
        }

        private static Footprint RandomFootprint(System.Random rng, int res)
        {
            int w, d;
            float height = 1f;
            double r = rng.NextDouble();
            if (r < 0.25)
            {
                int m = rng.Next(4, 9);                               // wall, 5-10 m
                if (rng.Next(2) == 0) { w = m; d = 1; } else { w = 1; d = m; }
            }
            else if (r < 0.45) { w = 2; d = 2; }                      // block
            else if (r < 0.60) { w = 1; d = 1; }                      // pillar
            else
            {
                height = 0.5f;                                        // crate
                if (rng.Next(2) == 0) { w = 2; d = 1; } else { w = 1; d = 2; }
            }
            return new Footprint(rng.Next(1, res - 1 - w), rng.Next(1, res - 1 - d), w, d, height);
        }

        /// <summary>The footprint and a one-cell ring around it are all floor.</summary>
        private static bool IsFree(Footprint f, float[] heights, int res)
        {
            for (int y = f.Y - 1; y <= f.Y + f.D; y++)
                for (int x = f.X - 1; x <= f.X + f.W; x++)
                {
                    if (x < 0 || y < 0 || x >= res || y >= res) return false;
                    if (heights[y * res + x] > 0f) return false;
                }
            return true;
        }

        /// <summary>A piece near the centre can meet its own twin.</summary>
        private static bool Touches(Footprint a, Footprint b)
            => a.X - 1 <= b.X + b.W && b.X - 1 <= a.X + a.W && a.Y - 1 <= b.Y + b.D && b.Y - 1 <= a.Y + a.D;

        private static void Fill(Footprint f, float[] heights, int res, float value)
        {
            for (int y = f.Y; y < f.Y + f.D; y++)
                for (int x = f.X; x < f.X + f.W; x++)
                    heights[y * res + x] = value;
        }

        private bool TooCloseToSpawnOrCentre(Footprint f)
        {
            float x0 = -Radius + ArenaLayout.CellSize * f.X, x1 = x0 + ArenaLayout.CellSize * f.W;
            float z0 = -Radius + ArenaLayout.CellSize * f.Y, z1 = z0 + ArenaLayout.CellSize * f.D;

            float Distance(Vector3 p)
            {
                float dx = Mathf.Max(x0 - p.x, 0f, p.x - x1);
                float dz = Mathf.Max(z0 - p.z, 0f, p.z - z1);
                return Mathf.Sqrt(dx * dx + dz * dz);
            }

            if (Distance(Vector3.zero) < CenterZoneRadius + 0.5f) return true;
            for (int team = 0; team < 2; team++)
                foreach (var slot in Layout.TeamSpawns[team])
                    if (Distance(slot - transform.position) < 3f) return true;
            return false;
        }

        private void CellOf(Vector3 world, out int x, out int y)
        {
            var local = world - transform.position;
            x = Mathf.FloorToInt((local.x + Radius) / ArenaLayout.CellSize);
            y = Mathf.FloorToInt((local.z + Radius) / ArenaLayout.CellSize);
        }

        /// <summary>ArenaLayout.HasLineOfSight's grid walk, run on the draft before it is committed.</summary>
        private bool CaptainsSeeEachOther(float[] heights, int res)
        {
            CellOf(Layout.TeamSpawns[0][0], out int x0, out int y0);
            CellOf(Layout.TeamSpawns[1][0], out int x1, out int y1);
            int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;
            while (x0 != x1 || y0 != y1)
            {
                int e2 = err * 2;
                if (e2 > -dy) { err -= dy; x0 += sx; }
                if (e2 < dx) { err += dx; y0 += sy; }
                if (x0 == x1 && y0 == y1) break;
                if (heights[y0 * res + x0] >= 0.9f) return false;
            }
            return true;
        }

        /// <summary>Every spawn slot reachable from team A's first over free cells, four-connected.</summary>
        private bool SpawnsConnected(float[] heights, int res)
        {
            var seen = new bool[res * res];
            var queue = new Queue<int>();
            CellOf(Layout.TeamSpawns[0][0], out int sx, out int sy);
            queue.Enqueue(sy * res + sx);
            seen[sy * res + sx] = true;
            while (queue.Count > 0)
            {
                int i = queue.Dequeue(), x = i % res, y = i / res;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (nx < 0 || ny < 0 || nx >= res || ny >= res) continue;
                    int n = ny * res + nx;
                    if (seen[n] || heights[n] > 0f) continue;
                    seen[n] = true;
                    queue.Enqueue(n);
                }
            }
            for (int team = 0; team < 2; team++)
                foreach (var slot in Layout.TeamSpawns[team])
                {
                    CellOf(slot, out int x, out int y);
                    if (!seen[y * res + x]) return false;
                }
            return true;
        }

        // ---- Divide (phase 5, the arena no world model has seen) -------------------

        /// <summary>
        /// A full-height wall splits the floor into two halves with three ways
        /// through: a 3 m corridor at each side and a 12 m gap around the centre.
        /// Spawns are Octagon's north/south pair, 34.7 m apart with a clear line
        /// through the gap, so contact still happens at once. Its two halves are
        /// staggered by one cell: the wall cannot sit on z = 0, which is a cell
        /// boundary, and stay point-symmetric as one piece. Same cell-centre rule
        /// as Warehouse (see BuildWarehouseCovers).
        /// </summary>
        private void BuildDivideCovers()
        {
            const float full = 2.6f, crouch = 1.1f;

            PlacePair("Divider", -11.5f, 0.625f, 0f, new Vector3(11f, full, 1.4f), 1f, 4);
            PlacePair("Block", 8.75f, 10f, 0f, new Vector3(2.4f, full, 2.4f), 1f, 1);
            PlacePair("Block", -12.5f, 8.75f, 0f, new Vector3(2.4f, full, 2.4f), 1f, 1);
            PlacePair("Pillar", -6.875f, 8.125f, 0f, new Vector3(1.6f, full, 1.6f), 1f, 1);

            var crate = new Vector3(2.6f, crouch, 1.2f);
            PlacePair("Crate", -5f, 3.125f, 0f, crate, 0.5f, 1);
            PlacePair("Crate", 0f, 11.875f, 0f, crate, 0.5f, 1);
            PlacePair("Crate", 15.625f, 7.5f, 90f, crate, 0.5f, 1);
            PlacePair("Crate", -18.125f, 3.75f, 90f, crate, 0.5f, 1);
        }

        /// <summary>
        /// A box at (x, z) and its twin at (-x, -z). A box is unchanged by a 180
        /// degree turn, so the twin keeps the same yaw and the pair -- anchors
        /// included -- is invariant under the rotation that swaps the spawns.
        /// </summary>
        private void PlacePair(string name, float x, float z, float yaw, Vector3 size, float height,
            int anchorsPerFace)
        {
            int index = _pairs++;
            PlaceBox($"{name}_{index}_A", new Vector3(x, 0f, z), yaw, size, height, anchorsPerFace);
            PlaceBox($"{name}_{index}_B", new Vector3(-x, 0f, -z), yaw, size, height, anchorsPerFace);
        }

        /// <summary>
        /// Like PlaceCover: standing spots on both broad faces, normals opposed.
        /// Long walls get several per face so an agent is not steered to one spot
        /// on a nine-metre shelf.
        /// </summary>
        private void PlaceBox(string name, Vector3 local, float yaw, Vector3 size, float height, int anchorsPerFace)
        {
            var rotation = Quaternion.Euler(0f, yaw, 0f);
            var center = transform.position + local;

            Block(name, local + Vector3.up * (size.y * 0.5f), rotation, size, _coverMaterial);
            Stamp(center, size, rotation, height);

            var normal = rotation * Vector3.forward;
            var along = rotation * Vector3.right;
            float standoff = size.z * 0.5f + 0.9f;

            for (int k = 0; k < anchorsPerFace; k++)
            {
                float t = anchorsPerFace == 1 ? 0f : (k / (float)(anchorsPerFace - 1) - 0.5f) * (size.x - 1f);
                var spot = center + along * t;
                Layout.CoverAnchors.Add(new CoverAnchor(spot + normal * standoff, normal, height));
                Layout.CoverAnchors.Add(new CoverAnchor(spot - normal * standoff, -normal, height));
            }
        }

        /// <summary>
        /// Opposite corners, slots spread across the diagonal, mirrored through the
        /// origin. Inset so the two captains start 37 m apart, inside the 40 m
        /// SightRange with a clear line through the centre. Neither brain searches
        /// for an enemy it has never seen: from the true corners (44 m) every round
        /// of a first test ran out the clock with no contact at all.
        /// </summary>
        private void BuildDiagonalSpawns()
        {
            var corner = new Vector3(Radius - 7f, 0f, Radius - 7f);
            var across = new Vector3(1f, 0f, -1f).normalized;

            for (int team = 0; team < 2; team++)
            {
                var slots = new Vector3[SpawnsPerTeam];
                float sign = team == 0 ? -1f : 1f;

                for (int i = 0; i < SpawnsPerTeam; i++)
                {
                    float lateral = (i - (SpawnsPerTeam - 1) * 0.5f) * SpawnSpread;
                    slots[i] = transform.position + sign * (corner + across * lateral);
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
