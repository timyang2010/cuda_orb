using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;
namespace TestEnumerator
{
   
    class Program
    {   
        static void stdout_recieved(object sender, DataReceivedEventArgs e)
        {
            if (!String.IsNullOrEmpty(e.Data))
            {
                Console.WriteLine(e.Data);
            }
        }

        static void StartProcess(string arguments)
        {
            ProcessStartInfo psi = new ProcessStartInfo()
            {
                RedirectStandardOutput = true,
                RedirectStandardInput = true,
                UseShellExecute = false,
                FileName = "Orb.exe",
                Arguments = arguments
            };

            Process proc = new Process() { StartInfo = psi };
            proc.OutputDataReceived += stdout_recieved;
            proc.Start();
            proc.BeginOutputReadLine();
            proc.WaitForExit();
            proc.Close();
        }


        [STAThread]
        static void Main(string[] args)
        {
            FolderBrowserDialog fbd = new FolderBrowserDialog();
            OpenFileDialog ofd = new OpenFileDialog() { Multiselect = true };
            while(ofd.ShowDialog() == DialogResult.OK)
            {
                StartProcess(string.Format("r {0}", string.Concat(ofd.FileNames.Select(fi => fi + " "))));
            }
            Console.WriteLine("\n------------------------------------\nProcess Complete");
            Console.ReadLine();
        }


    }
}
