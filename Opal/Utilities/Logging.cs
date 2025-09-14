using System.Collections.Concurrent;
using System.Threading;

namespace Opal.Utilities;

public static class Logging
{
    public enum LogLevel
    {
        LowDebug = 0,
        Debug = 1,
        HighDebug = 2,
        
        LowInfo = 3,
        Info = 4,
        HighInfo = 5,
        
        LowWarning = 6,
        Warning = 7,
        HighWarning = 8,
        
        LowError = 9,
        Error = 10,
        HighError = 11,
    }

    private static readonly ConcurrentQueue<(string name, int level, string message)> LogQueue = new();
    private static readonly AutoResetEvent LogEvent = new(false);
    private static readonly Thread LogThread;
    private static volatile bool _logThreadStarted = false;

    static Logging()
    {
        LogThread = new Thread(ProcessLogQueue) { IsBackground = true };
        LogThread.Start();
        _logThreadStarted = true;
    }

    public static void StandardLog(string name, int level, string message)
    {
        if (level < Core.LogLevel) return;
        if (Core.LogWhitelist.Count != 0 && !Core.LogWhitelist.Contains(name)) return;
        if (Core.LogBlacklist.Contains(name)) return;
        Console.WriteLine($"[{name}] [{level}] {message}");
    }

    public static void AsyncLog(string name, int level, string message)
    {
        if (level < Core.LogLevel) return;
        if (Core.LogWhitelist.Count != 0 && !Core.LogWhitelist.Contains(name)) return;
        if (Core.LogBlacklist.Contains(name)) return;
        LogQueue.Enqueue((name, level, message));
        LogEvent.Set();
    }

    private static void ProcessLogQueue()
    {
        while (true)
        {
            while (LogQueue.TryDequeue(out var log))
            {
                Console.WriteLine($"[{log.name}] [{log.level}] {log.message}");
            }
            LogEvent.WaitOne();
        }
    }
}