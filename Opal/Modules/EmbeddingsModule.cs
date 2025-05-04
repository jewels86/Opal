using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal.Modules
{
	public class EmbeddingsModule<T> : IModule
	{
		public int ID { get; private set; }
		public string Name { get; private set; }

		/// <summary>The number of buckets to store.</summary>
		public int K { get; private set; }
		/// <summary>The number of dimensions a single embedding contains.</summary>
		public int N { get; private set; }
		/// <summary>The embeddings stored in the module (bucket, (id, embedding)).</summary>
		public Dictionary<int, Dictionary<int, Embedding<T>>> Embeddings { get; private set; } = new();

		public EmbeddingsModule(int k, int n)
		{
			ID = Core.Register(this);
			Name = $"embeddings-{typeof(T).Name}";
			K = k;
			N = n;
		}

		public void Initialize() { }

		public void Receive(Packet packet)
		{

		}
	}

	public class Embedding<T>(int id, T data)
	{
		public int ID { get; private set } = id;
		public T Data { get; private set; } = data;
	}
}
