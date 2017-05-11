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
    public static class helper
    {
        public static void AddOnUI<T>(this ICollection<T> collection, T item)
        {
            Action<T> addMethod = collection.Add;
            System.Windows.Application.Current.Dispatcher.BeginInvoke(addMethod, item);
        }
    }
    public class Program : ViewModelBase
    {
       
        ObservableCollection<string> _messages = new ObservableCollection<string>();
        public ObservableCollection<string> messages
        {
            get { return _messages; }
        }

        void stdout_recieved(object sender, DataReceivedEventArgs e)
        {
            if (!String.IsNullOrEmpty(e.Data))
            { 
                helper.AddOnUI(messages, e.Data);
            }
        }

        void StartProcess(string arguments)
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

        static Dictionary<string, string> reverseMap;

        static Program()
        {
            reverseMap = new Dictionary<string, string>();
            reverseMap.Add("track", "c");
            reverseMap.Add("train", "t");
            reverseMap.Add("benchmark", "r");
            reverseMap.Add("match", "m");
        }
        public void run(string option,List<string> files)
        {
            string se = option;
            if (!reverseMap.ContainsValue(se))
            {
                if (reverseMap.ContainsKey(se))
                {
                    se = reverseMap[se];
                }
            }
            helper.AddOnUI(messages, "XXX");
            StartProcess(string.Format("{0} {1}", "t", string.Concat(files.Select(fi => fi + " "))));
        }


    }
}