param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$Version,
    [Parameter(Mandatory=$true)][string]$Description,
    [Parameter(Mandatory=$true)][string]$OriginalFilename
)
$ErrorActionPreference = 'Stop'
$file = Get-Item -LiteralPath $Path
if ($Version -notmatch '^\d+\.\d+\.\d+(\.\d+)?$') { throw 'Windows file version must have three or four numeric components.' }
if ($OriginalFilename -notmatch '^idletoken-[a-z-]+\.exe$') { throw 'Unexpected IdleToken executable name.' }
if ($Description -notmatch '^IdleToken(?: |$)' -or $Description -match '(?i)llama|ggml') {
    throw 'Windows file descriptions must use IdleToken product names without backend branding.'
}
$v = $file.VersionInfo
if ($v.ProductName -eq 'IdleToken' -and $v.CompanyName -eq 'IdleToken' -and
    $v.FileDescription -eq $Description -and $v.FileVersion -eq $Version -and
    $v.ProductVersion -eq $Version -and $v.OriginalFilename -eq $OriginalFilename) {
    Write-Output "WINDOWS_IDENTITY_OK: $OriginalFilename (already current)"
    exit 0
}
if ((Get-AuthenticodeSignature -LiteralPath $file.FullName).Status -ne 'NotSigned') {
    throw 'Resource changes must precede signing. Restage the unsigned build output.'
}
if (-not ('IdleTokenVersionResource' -as [type])) {
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class IdleTokenVersionResource {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    static extern IntPtr BeginUpdateResource(string file, bool deleteExisting);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool UpdateResource(IntPtr update, IntPtr type, IntPtr name, ushort language, byte[] data, uint size);
    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool EndUpdateResource(IntPtr update, bool discard);
    static void Align(BinaryWriter writer) { while ((writer.BaseStream.Position & 3) != 0) writer.Write((byte)0); }
    static byte[] Node(string key, ushort type, byte[] value, ushort valueLength, params byte[][] children) {
        using (var stream = new MemoryStream()) using (var writer = new BinaryWriter(stream)) {
            writer.Write((ushort)0); writer.Write(valueLength); writer.Write(type);
            writer.Write(Encoding.Unicode.GetBytes(key + "\0")); Align(writer);
            writer.Write(value);
            foreach (var child in children) { Align(writer); writer.Write(child); }
            long length = stream.Length;
            if (length > ushort.MaxValue) throw new InvalidOperationException("Version resource is too large");
            stream.Position = 0; writer.Write((ushort)length);
            return stream.ToArray();
        }
    }
    static byte[] Text(string key, string value) {
        return Node(key, 1, Encoding.Unicode.GetBytes(value + "\0"), checked((ushort)(value.Length + 1)));
    }
    public static void Stamp(string file, string version, string description, string original) {
        string[] parts = version.Split('.'); ushort[] numbers = new ushort[4];
        for (int i = 0; i < parts.Length; ++i) numbers[i] = ushort.Parse(parts[i]);
        uint ms = ((uint)numbers[0] << 16) | numbers[1], ls = ((uint)numbers[2] << 16) | numbers[3];
        byte[] fixedInfo;
        using (var stream = new MemoryStream()) using (var writer = new BinaryWriter(stream)) {
            foreach (uint value in new uint[] {0xFEEF04BD, 0x00010000, ms, ls, ms, ls, 0x3F, 0, 0x00040004, 1, 0, 0, 0}) writer.Write(value);
            fixedInfo = stream.ToArray();
        }
        byte[] empty = new byte[0];
        byte[] strings = Node("StringFileInfo", 1, empty, 0,
            Node("040904b0", 1, empty, 0,
                Text("CompanyName", "IdleToken"), Text("ProductName", "IdleToken"),
                Text("FileDescription", description), Text("FileVersion", version),
                Text("ProductVersion", version), Text("OriginalFilename", original),
                Text("InternalName", System.IO.Path.GetFileNameWithoutExtension(original)),
                Text("LegalCopyright", "Copyright IdleToken contributors. See bundled licenses.")));
        byte[] translations = Node("VarFileInfo", 1, empty, 0,
            Node("Translation", 0, new byte[] {0x09, 0x04, 0xb0, 0x04}, 4));
        byte[] resource = Node("VS_VERSION_INFO", 0, fixedInfo, 52, strings, translations);
        IntPtr update = BeginUpdateResource(file, false);
        if (update == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        if (!UpdateResource(update, (IntPtr)16, (IntPtr)1, 0x0409, resource, (uint)resource.Length)) {
            int error = Marshal.GetLastWin32Error(); EndUpdateResource(update, true); throw new Win32Exception(error);
        }
        if (!EndUpdateResource(update, false)) throw new Win32Exception(Marshal.GetLastWin32Error());
    }
}
'@
}
[IdleTokenVersionResource]::Stamp($file.FullName, $Version, $Description, $OriginalFilename)
$actual = [Diagnostics.FileVersionInfo]::GetVersionInfo($file.FullName)
if ($actual.ProductName -ne 'IdleToken' -or $actual.FileDescription -ne $Description -or
    $actual.FileVersion -ne $Version -or $actual.OriginalFilename -ne $OriginalFilename) {
    throw 'The written Windows version resource failed verification.'
}
Write-Output "WINDOWS_IDENTITY_OK: $OriginalFilename"
