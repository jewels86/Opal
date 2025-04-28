using MessagePack;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using static Opal.Utilities.ModuleUtilities;

namespace Opal.Modules.Memory
{

	public class EmbeddingNode
	{
		public int ID { get; set; }
		public double[] Vector { get; set; } = new double[128];
		public Dictionary<int, double> Associations { get; set; } = new();
		public Dictionary<string, string> Metadata { get; set; } = new();
	}

	public class EmbeddingEngineModule : IModule
	{
		public string ID => "memory:embedding-engine";
		public ConcurrentQueue<Packet> Input { get; } = new();
		public ConcurrentQueue<Packet> Output { get; } = new();

		public int EmbeddingSize { get; } = 128;
		public int EmbeddingAxisMax { get; } = 4;
		public ConcurrentBag<EmbeddingNode> Nodes { get; } = new();

		private Random _random = new();
		private int _nextNodeID = 1;
		private object _nextNodeIDLock = new object();


		public void Initialize(Context ctx)
		{
			ctx.Log(ID, 3, "Initializing EmbeddingEngineModule.");
			ctx.Add(this);
		}

		public void Main(Context ctx)
		{
			ctx.Log(ID, 3, "Starting main loop of EmbeddingEngineModule.");

			Action<Packet> main = (packet) =>
			{
				#region memory:embedding-engine->create 
				if (packet.Type == "memory:embedding-engine->create")
				{
					double[] vector = new double[EmbeddingSize];
					for (int i = 0; i < EmbeddingSize; i++)
					{
						vector[i] = _random.NextDouble() * EmbeddingAxisMax;
					}
					int newID;
					lock (_nextNodeIDLock) { newID = _nextNodeID++; }
					Nodes.Add(new EmbeddingNode { ID = newID, Vector = vector });
					ctx.Log(ID, 3, $"Created new embedding node with ID: {newID}");
					Output.Enqueue(new Packet
					{
						Type = "memory:embedding-engine->create-response",
						TargetID = packet.SourceID,
						SourceID = ID,
						Payload = newID,
						PayloadType = "int",
						PacketID = -packet.PacketID,
					});
				}
				#endregion
				#region memory:embedding-engine->associate
				else if (packet.Type == "memory:embedding-engine->associate")
				{
					if (TypeIs(packet.PayloadType, "(int, Dictionary<int, double>)"))
					{
						var payload = (ValueTuple<int, Dictionary<int, double>>)packet.Payload!;
						int nodeID = payload.Item1;
						Dictionary<int, double> associations = payload.Item2;
						EmbeddingNode? node = Nodes.FirstOrDefault(n => n.ID == nodeID);
						if (node == null)
						{
							ctx.Log(ID, 2, $"Node with ID {nodeID} not found.");
							return;
						}
						foreach (var kvp in associations)
						{
							node.Associations[kvp.Key] = kvp.Value;
						}
						ctx.Log(ID, 3, $"Associated {associations.Count} vectors with node {nodeID}-{SHAHash(node.Vector)} (hashed SHA256)");
						ctx.Log(ID, 3, $"Adjusting vector...");
						double[] averageVector = AverageVectors(Nodes.Where(n => associations.ContainsKey(n.ID)).Select(n => n.Vector).ToArray());
						averageVector = AverageVectors(new[] { node.Vector, averageVector });
						double[] normalized = NormalizeVector(averageVector);
						node.Vector = normalized;
						ctx.Log(ID, 3, $"New vector for node {nodeID}: {SHAHash(normalized)} (SHA256)");
						Output.Enqueue(new Packet
						{
							Type = "memory:embedding-engine->associate-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = true,
							PayloadType = "bool",
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for associate: {packet.PayloadType} (should be (int, Dictionary<int, double>)");
					}
				}
				#endregion
				#region memory:embedding-engine->find-similar
				else if (packet.Type == "memory:embedding-engine->find-similar")
				{
					if (TypeIs(packet.PayloadType, "int"))
					{
						int nodeID = (int)packet.Payload!;
						EmbeddingNode? node = Nodes.FirstOrDefault(n => n.ID == nodeID);
						if (node == null)
						{
							ctx.Log(ID, 2, $"Node with ID {nodeID} not found.");
							return;
						}
						double[] vector = node.Vector;
						List<(int ID, double Similarity)> similarities = new();
						foreach (var n in Nodes)
						{
							if (n.ID != nodeID)
							{
								if (packet.Data.TryGetValue("method", out string? method))
								{
									if (method == "cosine")
									{
										double similarity = CosineSimilarity(NormalizeVector(vector), NormalizeVector(n.Vector));
										similarities.Add((n.ID, similarity));
									}
									else if (method == "pearson")
									{
										double similarity = PearsonCorrelation(NormalizeVector(vector), NormalizeVector(n.Vector));
										similarities.Add((n.ID, similarity));
									}
									else if (method == "euclidean")
									{
										double similarity = EuclideanDistance(NormalizeVector(vector), NormalizeVector(n.Vector));
										similarities.Add((n.ID, similarity));
									}
								}
								else 
								{ 
									double similarity = CosineSimilarity(NormalizeVector(vector), NormalizeVector(n.Vector));
									similarities.Add((n.ID, similarity));
								}
							}
						}
						similarities = similarities.Where(s => s.Similarity > 0.5).ToList();
						ctx.Log(ID, 3, $"Found {similarities.Count} similar nodes to {nodeID}-{SHAHash(node.Vector)} (hashed SHA256)");
						if (similarities.Count != 0) 
							ctx.Log(ID, 3, $"Largest similarity: {similarities[0].Similarity} ({similarities[0].ID}), lowest similarity: {similarities.Last().Similarity} ({similarities.Last().ID})");
						Output.Enqueue(new Packet
						{
							Type = "memory:embedding-engine->find-similar-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = similarities,
							PayloadType = "List<(int ID, double Similarity)>",
							PacketID = -packet.PacketID,
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for find-similar: {packet.PayloadType} (should be int)");
					}
				}
				#endregion
				#region memory:embedding-engine->find-by-metadata-tag
				else if (packet.Type == "memory:embedding-engine->find-by-metadata-tag")
				{
					if (TypeIs(packet.PayloadType, "(string, string)"))
					{
						var payload = (ValueTuple<string, string>)packet.Payload!;
						string tag = payload.Item1;
						string value = payload.Item2;
						List<EmbeddingNode> foundNodes = Nodes.Where(n => n.Metadata.ContainsKey(tag) && n.Metadata[tag].Equals(value)).ToList();
						ctx.Log(ID, 3, $"Found {foundNodes.Count} nodes with metadata {tag}: {value}");
						Output.Enqueue(new Packet
						{
							Type = "memory:embedding-engine->find-by-metadata-tag-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = foundNodes,
							PayloadType = "List<EmbeddingNode>",
							PacketID = -packet.PacketID,
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for find-by-metadata: {packet.PayloadType} (should be (string, object))");
					}
				}
				#endregion
				#region memory:embedding-engine->add-metadata
				else if (packet.Type == "memory:embedding-engine->add-metadata")
				{
					if (TypeIs(packet.PayloadType, "(int, string, string)"))
					{
						var payload = (ValueTuple<int, string, string>)packet.Payload!;
						int nodeID = payload.Item1;
						string tag = payload.Item2;
						string value = payload.Item3;
						EmbeddingNode? node = Nodes.FirstOrDefault(n => n.ID == nodeID);
						if (node == null)
						{
							ctx.Log(ID, 2, $"Node with ID {nodeID} not found.");
							return;
						}
						node.Metadata[tag] = value;
						ctx.Log(ID, 3, $"Added metadata {tag}: {value} to node {nodeID}");
						Output.Enqueue(new Packet
						{
							Type = "memory:embedding-engine->add-metadata-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = true,
							PayloadType = "bool",
							PacketID = -packet.PacketID,
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for add-metadata: {packet.PayloadType} (should be (int, string, object))");
					}
				}
				#endregion
				#region memory:embedding-engine->get-id
				else if (packet.Type == "memory:embedding-engine->get-id")
				{
					if (TypeIs(packet.PayloadType, "string"))
					{
						string word = (string)packet.Payload!;
						EmbeddingNode? node = Nodes.FirstOrDefault(n => n.Metadata.ContainsKey("word") && n.Metadata["word"] == word);
						if (node == null)
						{
							ctx.Log(ID, 2, $"Node with word {word} not found.");
							return;
						}
						ctx.Log(ID, 3, $"Found node with word {word}: {node.ID}");
						Output.Enqueue(new Packet
						{
							Type = "memory:embedding-engine->get-id-response",
							TargetID = packet.SourceID,
							SourceID = ID,
							Payload = node.ID,
							PayloadType = "int",
							PacketID = -packet.PacketID,
						});
					}
					else
					{
						ctx.Log(ID, 2, $"Invalid payload type for get-id: {packet.PayloadType} (should be string)");
					}
				}
				#endregion
				else
				{
					ctx.Log(ID, 2, $"Unknown packet type: {packet.Type}");

				}
			};

			List<Task> tasks = [];

			while (ctx.ShouldNotExit())
			{
				CheckForInput(this, main, ref tasks);
			}

			Task.WaitAll(tasks.ToArray());
			ctx.Log(ID, 3, "Exiting main loop of EmbeddingEngineModule.");
		}

		private double[] AverageVectors(double[][] vectors)
		{
			double[] average = new double[EmbeddingSize];
			for (int i = 0; i < EmbeddingSize; i++)
			{
				double sum = vectors.Sum(v => v[i]);
				average[i] = sum / vectors.Length;
			}
			return average;
		}

		private double[] NormalizeVector(double[] vector)
		{
			double length = Math.Sqrt(vector.Sum(v => v * v));
			for (int i = 0; i < vector.Length; i++)
			{
				vector[i] /= length;
			}
			return vector;
		}

		private double CosineSimilarity(double[] vectorA, double[] vectorB)
		{
			double dotProduct = 0;
			double lengthA = 0;
			double lengthB = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				dotProduct += vectorA[i] * vectorB[i];
				lengthA += vectorA[i] * vectorA[i];
				lengthB += vectorB[i] * vectorB[i];
			}
			if (lengthA == 0 || lengthB == 0)
			{
				return 0;
			}
			return dotProduct / (Math.Sqrt(lengthA) * Math.Sqrt(lengthB));
		}
		private double PearsonCorrelation(double[] vectorA, double[] vectorB)
		{
			double sumA = vectorA.Sum();
			double sumB = vectorB.Sum();
			double sumASquared = vectorA.Sum(v => v * v);
			double sumBSquared = vectorB.Sum(v => v * v);
			double sumProduct = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sumProduct += vectorA[i] * vectorB[i];
			}
			int n = vectorA.Length;
			double numerator = n * sumProduct - sumA * sumB;
			double denominator = Math.Sqrt((n * sumASquared - sumA * sumA) * (n * sumBSquared - sumB * sumB));
			if (denominator == 0)
			{
				return 0;
			}
			return numerator / denominator;
		}
		private double EuclideanDistance(double[] vectorA, double[] vectorB)
		{
			double sum = 0;
			for (int i = 0; i < vectorA.Length; i++)
			{
				sum += (vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i]);
			}
			return Math.Sqrt(sum);
		}
	}
}
