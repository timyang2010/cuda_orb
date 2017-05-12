using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Windows.Forms;
using GalaSoft.MvvmLight;
using System.Windows;
using System.Collections.ObjectModel;
namespace loader.ViewModel
{
 
    public static class Program
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
                RedirectStandardInput = false,
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

        public static void run(string option,List<string> files)
        {       
            StartProcess(string.Format("{0} {1}",option, string.Concat(files.Select(fi => fi + " "))));
        }


    }
}