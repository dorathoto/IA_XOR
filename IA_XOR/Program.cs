using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.Neuro;
using Accord.Neuro.ActivationFunctions;
using Accord.Neuro.Learning;
using Accord.Math;

namespace IA_XOR
{
    internal class Program
    {
       
        
        static void Main(string[] args)
        {
            double[][] input =
               {
                    new double[] {0, 0},
                    new double[] {0, 1},
                    new double[] {1, 0},
                    new double[] {1, 1}
                };

            double[][] output =
                {
                    new double[] {0},
                    new double[] {1},
                    new double[] {1},
                    new double[] {0}
                };
            
            ActivationNetwork network = new ActivationNetwork(
    new SigmoidFunction(2),
    2, // two inputs
    2, // two neurons in the first (hidden) layer
    1); // one neuron in the output layer

            var teacher = new BackPropagationLearning(network);

            int iteration = 0;
            double error = double.PositiveInfinity;


            while (error > 1e-6)
            {
                error = teacher.RunEpoch(input, output);
                iteration++;
                if (iteration % 1000 == 0)
                {
                    Console.WriteLine("Iteration: {0}, Error: {1}", iteration, error);
                }
            }

            double[] prediction = network.Compute(new double[] { 0, 0 });

            Console.WriteLine("Prediction for [0, 0]: " + prediction[0]);
        }
    }
}
