using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MessagePack;

namespace Opal.Utilities
{
	public class SemanticNetworkNode<T>
	{
		public T ID { get; set; }
		public List<SemanticNetworkConnection<T>> Connections { get; set; } = [];

		public SemanticNetworkNode(T id)
		{
			ID = id;
		}

		public int Connect(SemanticNetworkNode<T> node, float weight)
		{
			SemanticNetworkConnection<T> connection = new(this, node, weight);
			Connections.Add(connection);
			node.Connections.Add(connection);
			return Connections.Count - 1;
		}

		public float Weight()
		{
			return Connections.Sum(c => c.Weight);
		}

		public SemanticNetworkNodeData<T> Export()
		{
			SemanticNetworkNodeData<T> data = new()
			{
				ID = ID,
				Connections = [.. Connections.Select(c => c.Export())]
			};
			return data;
		}
	}

	public class SemanticNetworkNodeData<T>
	{
		[Key(0)]
		public required T ID { get; set; }
		[Key(1)]
		public required List<SemanticNetworkConnectionData<T>> Connections { get; set; }
	}

	public class SemanticNetworkConnection<T>
	{
		public SemanticNetworkNode<T> A { get; set; }
		public SemanticNetworkNode<T> B { get; set; }
		public float Weight { get; set; }

		public SemanticNetworkConnection(SemanticNetworkNode<T> a, SemanticNetworkNode<T> b, float weight)
		{
			A = a;
			B = b;
			Weight = weight;
		}

		public SemanticNetworkConnectionData<T> Export()
		{
			SemanticNetworkConnectionData<T> data = new()
			{
				A = A.ID,
				B = B.ID,
				Weight = Weight
			};
			return data;
		}
	}

	public class SemanticNetworkConnectionData<T>
	{
		[Key(0)]
		public required T A { get; set; }
		[Key(1)]
		public required T B { get; set; }
		[Key(2)]
		public required float Weight { get; set; }
	}

	public class SemanticNetwork<T> where T : notnull
	{
		public Dictionary<T, SemanticNetworkNode<T>> Nodes { get; set; } = [];
		
		public SemanticNetworkNode<T> GetOrCreateNode(T id)
		{
			if (Nodes.ContainsKey(id)) { return Nodes[id]; }
			else
			{
				SemanticNetworkNode<T> node = new(id);
				Nodes[id] = node;
				return node;
			}
		}

		public int Connect(T idA,  T idB, float weight)
		{
			SemanticNetworkNode<T> a = GetOrCreateNode(idA);
			SemanticNetworkNode<T> b = GetOrCreateNode(idB);
			return a.Connect(b, weight);
		}
		public bool Disconnect(T idA, T idB)
		{
			if (Nodes.ContainsKey(idA) && Nodes.ContainsKey(idB))
			{
				SemanticNetworkNode<T> a = Nodes[idA];
				SemanticNetworkNode<T> b = Nodes[idB];
				SemanticNetworkConnection<T>? connection = a.Connections.FirstOrDefault(c => c.B == b);
				if (connection is not null)
				{
					a.Connections.Remove(connection);
					b.Connections.Remove(connection);
					return true;
				}
			}
			return false;
		}
	}
}
