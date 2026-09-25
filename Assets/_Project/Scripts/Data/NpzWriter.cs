using System;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

namespace JevNpcBrain.Data
{
    /// <summary>
    /// Writes NumPy .npz archives -- a zip of .npy files, one per array -- so the
    /// Python side reads a shard with a bare np.load() and there is no custom
    /// reader to get wrong. Arrays are C-order, little-endian, deflate-compressed.
    /// </summary>
    public sealed class NpzWriter : IDisposable
    {
        private readonly FileStream _file;
        private readonly ZipArchive _zip;
        private readonly byte[] _chunk = new byte[1 << 20];

        public NpzWriter(string path)
        {
            // Every writer below copies raw memory; a big-endian host would produce
            // files NumPy reads as garbage without complaint.
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("NpzWriter assumes a little-endian host.");

            _file = File.Create(path);
            _zip = new ZipArchive(_file, ZipArchiveMode.Create);
        }

        /// <summary>Half-precision floats, given as their IEEE binary16 bit patterns.</summary>
        public void WriteHalf(string name, ushort[] bits, params int[] shape) => Write(name, "<f2", bits, 2, shape);

        public void Write(string name, float[] data, params int[] shape) => Write(name, "<f4", data, 4, shape);
        public void Write(string name, byte[] data, params int[] shape) => Write(name, "|u1", data, 1, shape);
        public void Write(string name, sbyte[] data, params int[] shape) => Write(name, "|i1", data, 1, shape);
        public void Write(string name, int[] data, params int[] shape) => Write(name, "<i4", data, 4, shape);
        public void Write(string name, long[] data, params int[] shape) => Write(name, "<i8", data, 8, shape);

        /// <summary>
        /// Writes the first product(shape) elements of <paramref name="data"/>, which
        /// may be a larger, growable buffer.
        /// </summary>
        private void Write(string name, string descr, Array data, int itemSize, int[] shape)
        {
            long elements = 1;
            for (int i = 0; i < shape.Length; i++) elements *= shape[i];
            if (elements > data.Length)
                throw new ArgumentException($"{name}: shape needs {elements} elements, buffer has {data.Length}.");

            long bytes = elements * itemSize;
            var entry = _zip.CreateEntry(name + ".npy", CompressionLevel.Fastest);

            using (var stream = entry.Open())
            {
                var header = Header(descr, shape);
                stream.Write(header, 0, header.Length);

                for (long offset = 0; offset < bytes; offset += _chunk.Length)
                {
                    int count = (int)Math.Min(_chunk.Length, bytes - offset);
                    Buffer.BlockCopy(data, (int)offset, _chunk, 0, count);
                    stream.Write(_chunk, 0, count);
                }
            }
        }

        /// <summary>
        /// .npy version 1.0: magic, version, little-endian header length, then a
        /// Python dict literal padded with spaces to a 64-byte boundary and closed
        /// by a newline -- the layout NumPy itself writes.
        /// </summary>
        private static byte[] Header(string descr, int[] shape)
        {
            var dims = new string[shape.Length];
            for (int i = 0; i < shape.Length; i++) dims[i] = shape[i].ToString(CultureInfo.InvariantCulture);

            // A one-element tuple needs its trailing comma in Python.
            string tuple = shape.Length == 1 ? "(" + dims[0] + ",)" : "(" + string.Join(", ", dims) + ")";
            string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + tuple + ", }";

            const int preamble = 10;
            int unpadded = preamble + dict.Length + 1;
            int total = (unpadded + 63) / 64 * 64;
            string text = dict + new string(' ', total - unpadded) + "\n";

            var header = new byte[preamble + text.Length];
            header[0] = 0x93;
            Encoding.ASCII.GetBytes("NUMPY", 0, 5, header, 1);
            header[6] = 1;
            header[7] = 0;
            header[8] = (byte)(text.Length & 0xff);
            header[9] = (byte)(text.Length >> 8);
            Encoding.ASCII.GetBytes(text, 0, text.Length, header, preamble);
            return header;
        }

        public void Dispose()
        {
            _zip.Dispose();
            _file.Dispose();
        }
    }

    /// <summary>
    /// float to IEEE 754 binary16, round-to-nearest-even, with subnormals, infinity
    /// and NaN handled. Managed on purpose: Mathf.FloatToHalf crosses into native
    /// code once per value, and one shard is tens of millions of values.
    /// </summary>
    public static class Half16
    {
        [StructLayout(LayoutKind.Explicit)]
        private struct Bits
        {
            [FieldOffset(0)] public float Float;
            [FieldOffset(0)] public uint UInt;
        }

        private const uint MagicBits = 126u << 23; // 0.5f

        public static ushort FromFloat(float value)
        {
            uint x = new Bits { Float = value }.UInt;
            uint sign = x & 0x80000000u;
            x ^= sign;

            uint result;
            if (x >= 0x47800000u)
            {
                // Too big for half: infinity. NaN stays a (quiet) NaN.
                result = x > 0x7f800000u ? 0x7e00u : 0x7c00u;
            }
            else if (x < 0x38800000u)
            {
                // Subnormal or zero. Adding 0.5f lines the ten mantissa bits up at
                // the bottom of the float and lets the FPU do the rounding.
                float sum = new Bits { UInt = x }.Float + new Bits { UInt = MagicBits }.Float;
                result = new Bits { Float = sum }.UInt - MagicBits;
            }
            else
            {
                uint mantissaOdd = (x >> 13) & 1u;
                x += unchecked((uint)((15 - 127) << 23)) + 0xfffu;
                x += mantissaOdd;
                result = x >> 13;
            }

            return (ushort)(result | (sign >> 16));
        }
    }
}
