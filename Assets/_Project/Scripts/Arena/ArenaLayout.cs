using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Arena
{
    /// <summary>
    /// A cover piece with the direction it actually shelters you from.
    /// A cover is only cover relative to where the shooter is -- storing the
    /// normal lets the sensor score that honestly instead of treating every
    /// block as omnidirectional protection.
    /// </summary>
    public readonly struct CoverAnchor
    {
        /// <summary>Standing spot beside the cover, not the cover's own centre.</summary>
        public readonly Vector3 Position;

        /// <summary>Outward direction the cover blocks fire from.</summary>
        public readonly Vector3 Normal;

        /// <summary>1 for full-height, 0.5 for crouch-height.</summary>
        public readonly float Height;

        public CoverAnchor(Vector3 position, Vector3 normal, float height)
        {
            Position = position;
            Normal = normal;
            Height = height;
        }
    }

    /// <summary>
    /// Everything the AI layers need to know about the arena, computed once at
    /// build time so no brain ever raycasts the static world at 10 Hz.
    ///
    /// Deliberately not a MonoBehaviour: the layout outlives any particular scene
    /// object, and the cross-arena generalisation test in phase 5 swaps it wholesale.
    /// </summary>
    public sealed class ArenaLayout
    {
        /// <summary>The arena the current match is being played in.</summary>
        public static ArenaLayout Current { get; internal set; }

        /// <summary>Matches TacticalObservation.CellSize so no resampling is needed.</summary>
        public const float CellSize = 1.25f;

        public readonly string Name;
        public readonly float Radius;
        public readonly int Resolution;
        public readonly Vector3 Origin;

        /// <summary>Row-major [y * Resolution + x]. 0 open, 0.5 crouch cover, 1 full blocker.</summary>
        public readonly float[] Occupancy;

        public readonly List<CoverAnchor> CoverAnchors = new List<CoverAnchor>();

        /// <summary>Spawn points per team. Index 0 and 1 are the two teams.</summary>
        public readonly Vector3[][] TeamSpawns = new Vector3[2][];

        public readonly Vector3 CenterPosition;
        public readonly float CenterRadius;

        public ArenaLayout(string name, float radius, Vector3 origin, float centerRadius)
        {
            Name = name;
            Radius = radius;
            Origin = origin;
            CenterPosition = origin;
            CenterRadius = centerRadius;

            // 40 m across at 1.25 m per cell lands exactly on 32 -- the same window
            // the observation grid uses, which keeps the sensor a straight copy.
            Resolution = Mathf.RoundToInt(radius * 2f / CellSize);
            Occupancy = new float[Resolution * Resolution];
        }

        public bool TryWorldToCell(Vector3 world, out int x, out int y)
        {
            var local = world - Origin;
            x = Mathf.FloorToInt((local.x + Radius) / CellSize);
            y = Mathf.FloorToInt((local.z + Radius) / CellSize);
            return x >= 0 && x < Resolution && y >= 0 && y < Resolution;
        }

        public Vector3 CellToWorld(int x, int y) => Origin + new Vector3(
            (x + 0.5f) * CellSize - Radius,
            0f,
            (y + 0.5f) * CellSize - Radius);

        /// <summary>
        /// Obstacle height at a world position. Anything outside the arena reads as
        /// a full blocker -- which is true, it is the perimeter wall, and it means
        /// an agent near the edge correctly sees its back as protected.
        /// </summary>
        public float SampleHeight(Vector3 world)
            => TryWorldToCell(world, out int x, out int y) ? Occupancy[y * Resolution + x] : 1f;

        public float SampleHeight(int x, int y)
            => x >= 0 && x < Resolution && y >= 0 && y < Resolution
                ? Occupancy[y * Resolution + x]
                : 1f;

        internal void SetHeight(int x, int y, float height)
        {
            if (x < 0 || x >= Resolution || y < 0 || y >= Resolution) return;
            int i = y * Resolution + x;
            if (height > Occupancy[i]) Occupancy[i] = height;
        }

        /// <summary>
        /// Grid-walk line of sight against the static occupancy map. Crouch-height
        /// cover blocks a standing shot only partially, so the caller decides what
        /// counts -- pass blockingHeight 1f for "can I see over everything but walls".
        /// </summary>
        public bool HasLineOfSight(Vector3 from, Vector3 to, float blockingHeight = 0.9f)
        {
            if (!TryWorldToCell(from, out int x0, out int y0)) return false;
            if (!TryWorldToCell(to, out int x1, out int y1)) return false;

            int dx = Mathf.Abs(x1 - x0), dy = Mathf.Abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            while (x0 != x1 || y0 != y1)
            {
                int e2 = err * 2;
                if (e2 > -dy) { err -= dy; x0 += sx; }
                if (e2 < dx) { err += dx; y0 += sy; }

                if (x0 == x1 && y0 == y1) break;
                if (SampleHeight(x0, y0) >= blockingHeight) return false;
            }

            return true;
        }

        public bool IsInCenterZone(Vector3 world)
        {
            var d = world - CenterPosition;
            d.y = 0f;
            return d.sqrMagnitude <= CenterRadius * CenterRadius;
        }
    }
}
