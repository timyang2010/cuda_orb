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
                Console.Write('\r'+e.Data+"                                  ");
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
            const string metafilename = "meta.txt";
            FolderBrowserDialog fbd = new FolderBrowserDialog();
            OpenFileDialog ofd = new OpenFileDialog() { Multiselect = true };

            while(ofd.ShowDialog() == DialogResult.OK)
            {
                using (StreamWriter file = new StreamWriter(metafilename))
                {
                    foreach (var fi in ofd.FileNames)
                    {
                        file.WriteLine(fi);
                    }
                }
                StartProcess(string.Format("c {0}",metafilename));
            }
          

            Console.WriteLine("\n------------------------------------\nProcess Complete");
            //File.Delete(metafilename);
            Console.ReadLine();
        }


    }
}
