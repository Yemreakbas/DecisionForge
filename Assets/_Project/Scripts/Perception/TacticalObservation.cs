using UnityEngine;

namespace JevNpcBrain.Perception
{
    using Core;

    /// <summary>
    /// One agent's view of the world at one tactical tick.
    ///
    /// Both the hand-authored baseline and the learned JEV brain are handed the
    /// exact same instance of this type. That is the fairness contract of the whole
    /// experiment: the only thing that differs between the two teams is the
    /// function that maps this observation to a <see cref="TacticalIntent"/>.
    ///
    /// The grid is agent-centred but world-axis-aligned; heading lives in
    /// <see cref="Self"/> as a sin/cos pair. Rotating the grid to the agent's facing
    /// would likely generalise better but costs a resample every tick -- revisit
    /// once the baseline numbers are in.
    ///
    /// Layout is CHW so it can be handed to ONNX as [1, C, H, W] without a copy.
    /// </summary>
    public sealed class TacticalObservation
    {
        public const int GridSize = 32;
        public const float CellSize = 1.25f;          // 32 * 1.25 = 40 m, one arena width
        public const int ChannelCount = 7;
        public const int CellsPerChannel = GridSize * GridSize;
        public const int GridLength = ChannelCount * CellsPerChannel;

        /// <summary>Grid index of the agent itself.</summary>
        public const int CenterCell = GridSize / 2;

        public readonly float[] Grid = new float[GridLength];
        public readonly float[] SelfVector = new float[SelfState.Length];

        public SelfState Self;

        /// <summary>
        /// Cheap scalar features derived from the grid. The baseline utility brain
        /// consumes these; the JEV brain ignores them and reads the raw grid. Same
        /// source data either way -- this is a convenience, not extra information.
        /// </summary>
        public ObservationSummary Summary;

        /// <summary>World position the grid is centred on.</summary>
        public Vector3 Origin;

        public float Timestamp;

        public void Clear()
        {
            System.Array.Clear(Grid, 0, Grid.Length);
            Summary = default;
        }

        public static int Index(ObservationChannel channel, int x, int y)
            => (int)channel * CellsPerChannel + y * GridSize + x;

        public float Get(ObservationChannel channel, int x, int y)
            => Grid[Index(channel, x, y)];

        public void Set(ObservationChannel channel, int x, int y, float value)
            => Grid[Index(channel, x, y)] = value;

        public void Accumulate(ObservationChannel channel, int x, int y, float value)
            => Grid[Index(channel, x, y)] += value;

        public static bool InBounds(int x, int y)
            => x >= 0 && x < GridSize && y >= 0 && y < GridSize;

        /// <summary>World position -> grid cell. Returns false if outside the window.</summary>
        public bool TryWorldToCell(Vector3 world, out int x, out int y)
        {
            var local = world - Origin;
            x = CenterCell + Mathf.RoundToInt(local.x / CellSize);
            y = CenterCell + Mathf.RoundToInt(local.z / CellSize);
            return InBounds(x, y);
        }

        public Vector3 CellToWorld(int x, int y) => Origin + new Vector3(
            (x - CenterCell) * CellSize,
            0f,
            (y - CenterCell) * CellSize);

        /// <summary>Flattens self state into the fixed-length array fed to the model.</summary>
        public void FinalizeSelfVector()
        {
            Self.WriteTo(SelfVector);
        }
    }

    /// <summary>
    /// Scalar digest of the grid, computed once per tick by the sensor so that
    /// neither brain pays for it twice.
    /// </summary>
    public struct ObservationSummary
    {
        public float CoverQualityHere;
        public float ThreatExposureHere;

        public float BestCoverQuality;
        public float BestCoverDistance;        // metres
        public Vector3 BestCoverPosition;
        public bool HasBestCover;

        public float FlankCoverQuality;        // best cover that is off the threat axis
        public Vector3 FlankCoverPosition;
        public bool HasFlankCover;

        public float EnemyBeliefMass;          // 0 = no idea where anyone is
        public Vector3 EnemyBeliefCentroid;
        public float EnemyBeliefDistance;      // metres
        public float EnemyBeliefSpread;        // high = belief is smeared, we are lost

        public float NearestAllyDistance;
        public bool HasAlly;

        public float CenterPressure;           // believed enemy mass near the contest zone
        public float DistanceToCenter;
    }
}
