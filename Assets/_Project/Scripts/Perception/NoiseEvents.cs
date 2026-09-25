using System.Collections.Generic;
using UnityEngine;

namespace JevNpcBrain.Perception
{
    /// <summary>
    /// Shared ring buffer of things that made a sound. Gunfire is the main one.
    ///
    /// This is how an agent learns about an enemy it cannot see, which is what
    /// makes the EnemyBelief channel interesting rather than a delayed copy of
    /// ground truth.
    /// </summary>
    public static class NoiseEvents
    {
        public struct Noise
        {
            public Vector3 Position;
            public float Loudness;
            public float Time;
            public int SourceTeam;
        }

        private const int Capacity = 64;
        private static readonly Noise[] Buffer = new Noise[Capacity];
        private static int _head;

        /// <summary>How long a noise stays worth listening to.</summary>
        public const float Lifetime = 4f;

        public static void Report(Vector3 position, float loudness, int sourceTeam, float time)
        {
            Buffer[_head] = new Noise
            {
                Position = position,
                Loudness = loudness,
                Time = time,
                SourceTeam = sourceTeam
            };
            _head = (_head + 1) % Capacity;
        }

        public static void Clear()
        {
            for (int i = 0; i < Capacity; i++) Buffer[i] = default;
            _head = 0;
        }

        /// <summary>Noises still audible to the given team, newest first.</summary>
        public static void Collect(int listenerTeam, float now, List<Noise> into)
        {
            into.Clear();
            for (int i = 0; i < Capacity; i++)
            {
                var n = Buffer[i];
                if (n.Loudness <= 0f) continue;
                if (now - n.Time > Lifetime) continue;
                if (n.SourceTeam == listenerTeam) continue;
                into.Add(n);
            }
        }
    }
}
