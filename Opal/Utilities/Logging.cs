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

    /// <summary>
    /// Log levels that can be added to existing log levels for finer control.
    /// </summary>
    /// <remarks>
    /// Generally:
    /// <ul>
    ///     <li>Modules will log at LowBaseline for the beginning of routine operations or during them. They will log at Baseline for the end of routine operations.</li>
    ///     <li>Operations that are not routine but also not urgent will log at Baseline.</li>
    ///     <li>Operations that are important and should be looked at soon will log at Urgent.</li>
    ///     <li>Errors will increase the log level by one step; thus, a LowBaseline operation that encounters an error will log at Baseline, and an Urgent operation that encounters an error will log at HighUrgent.</li>
    /// </ul>
    /// Some operations will not log at all; this is not because the baseline is too low or because they are Negligible, but because they are not logged for other reasons (e.g., they would create excessive logging).
    /// </remarks>
    public enum AddedLogLevel
    {
        Negligible = -3,
        Unimportant = -2,
        LowBaseline = -1,
        
        Baseline = 0,
        
        HighBaseline = 1,
        Urgent = 2,
        HighUrgent = 3,
    }
    
    public static LogLevel CurrentLevel { get; set; } = LogLevel.Info;
    public static List<string> Blacklist { get; } = new();

    private static readonly ConcurrentQueue<(string name, LogLevel level, string message)> LogQueue = new();
    private static readonly AutoResetEvent LogEvent = new(false);
    private static readonly Thread LogThread;
    private static volatile bool _logThreadStarted = false;

    static Logging()
    {
        LogThread = new Thread(ProcessLogQueue) { IsBackground = true };
        LogThread.Start();
        _logThreadStarted = true;
    }

    public static void StandardLog(string name, LogLevel level, string message)
    {
        if (level < CurrentLevel) return;
        if (Blacklist.Contains(name)) return;
        Console.WriteLine($"[{name}] [{level}] {message}");
    }

    public static void AsyncLog(string name, LogLevel level, string message)
    {
        if (level < CurrentLevel) return;
        if (Blacklist.Contains(name)) return;
        LogQueue.Enqueue((name, level, message));
        LogEvent.Set();
    }

    public static void Log(string name, LogLevel level, string message, bool async = true)
    {
        if (async && _logThreadStarted)
            AsyncLog(name, level, message);
        else
            StandardLog(name, level, message);
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

public static class LogLevelExtensions
{
    public static Logging.LogLevel Add(this Logging.LogLevel level, Logging.AddedLogLevel addedLevel) =>
        (Logging.LogLevel)((int)level + (int)addedLevel);
}