using System.Collections.Concurrent;
using System.Threading;

namespace Opal.Utilities;

public static class Logging
{
    public enum LogLevel
    {
        Debug, HighDebug,
        Info, HighInfo, 
        Warning, HighWarning,
        Error, HighError
    }
    
    public static LogLevel CurrentLevel { get; set; } = LogLevel.Info;
    public static List<string> Blacklist { get; } = [];

    public static void Log(string name, LogLevel level, string message)
    {
        if (level < CurrentLevel) return;
        if (Blacklist.Contains(name)) return;
        Console.WriteLine($"[{name}] [{level}] {message}");
    }
    
    public static void AddToBlacklist(string name) => Blacklist.Add(name);
    public static void SetLogLevel(LogLevel level) => CurrentLevel = level;
}