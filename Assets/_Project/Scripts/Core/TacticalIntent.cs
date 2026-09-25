namespace JevNpcBrain.Core
{
    /// <summary>
    /// The shared action space. Every brain -- hand-authored or learned -- emits
    /// exactly one of these per tactical tick, and the reflex layer turns it into
    /// movement and aiming. Keeping the space small and named (rather than raw
    /// velocity) keeps the learning problem tractable and, just as importantly,
    /// makes a decision log readable by a human.
    /// </summary>
    public enum TacticalIntent : byte
    {
        Hold = 0,
        PushCoverForward = 1,
        PushCoverFlank = 2,
        Retreat = 3,
        FlankLeft = 4,
        FlankRight = 5,
        Suppress = 6,
        Peek = 7,
        RegroupAlly = 8,
        ContestCenter = 9
    }

    public static class TacticalIntents
    {
        public const int Count = 10;

        public static readonly TacticalIntent[] All =
        {
            TacticalIntent.Hold,
            TacticalIntent.PushCoverForward,
            TacticalIntent.PushCoverFlank,
            TacticalIntent.Retreat,
            TacticalIntent.FlankLeft,
            TacticalIntent.FlankRight,
            TacticalIntent.Suppress,
            TacticalIntent.Peek,
            TacticalIntent.RegroupAlly,
            TacticalIntent.ContestCenter
        };

        /// <summary>Short label used by the spectator overlay and the decision log.</summary>
        public static string Label(TacticalIntent intent) => intent switch
        {
            TacticalIntent.Hold => "HOLD",
            TacticalIntent.PushCoverForward => "PUSH",
            TacticalIntent.PushCoverFlank => "PUSH-FLANK",
            TacticalIntent.Retreat => "RETREAT",
            TacticalIntent.FlankLeft => "FLANK-L",
            TacticalIntent.FlankRight => "FLANK-R",
            TacticalIntent.Suppress => "SUPPRESS",
            TacticalIntent.Peek => "PEEK",
            TacticalIntent.RegroupAlly => "REGROUP",
            TacticalIntent.ContestCenter => "CONTEST",
            _ => "?"
        };
    }
}
