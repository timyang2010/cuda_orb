using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Forms;
using System.Linq;
using System.Windows.Media;


namespace loader.ViewModel
{
    using ConsoleControl.WPF;
    using System.Collections.Generic;

    /// <summary>
    /// This class contains properties that the main View can data bind to.
    /// <para>
    /// Use the <strong>mvvminpc</strong> snippet to add bindable properties to this ViewModel.
    /// </para>
    /// <para>
    /// You can also use Blend to data bind with the tool's support.
    /// </para>
    /// <para>
    /// See http://www.galasoft.ch/mvvm
    /// </para>
    /// </summary>
    public class MainViewModel : ViewModelBase
    {

        /// 

        public ObservableCollection<ImageSource> Sources { get { return _Sources; } }
        private ObservableCollection<ImageSource> _Sources = new ObservableCollection<ImageSource>();

        private ConsoleControl _cc;
        public ConsoleControl cc
        {
            get { return _cc; }
            set
            {
                _cc = value;
                RaisePropertyChanged("cc");
            }
        }
        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public MainViewModel()
        {
            cc = new ConsoleControl()
            {
                Margin = new Thickness(-2),
                FontSize = 17

            };
            cc.WriteOutput("Initialized", Colors.Red);
            cc.ClearOutput();
        }
        string _Selected;
        public string Selected
        {
            get { return _Selected; }
            set
            {
                _Selected = value;
                RaisePropertyChanged("Selected");

            }
        }
        private RelayCommand _LoadCommand;
        public RelayCommand LoadCommand
        {
            get
            {
                if (_LoadCommand == null) _LoadCommand = new RelayCommand(() =>
                {
                    OpenFileDialog ofd = new OpenFileDialog() { Multiselect = true };
                    if (ofd.ShowDialog() == DialogResult.OK)
                    {

                        foreach (var f in ofd.FileNames)
                        {
                            Sources.Add(new ImageSource(f));
                        }
                    }
                });
                return _LoadCommand;
            }
        }

        public void run(string option, List<string> files)
        {
            cc.StartProcess("Orb.exe", string.Format("{0} {1}", option, string.Concat(files.Select(fi => fi + " "))));
        }

        private RelayCommand<string> _ExecuteCommand;
        public RelayCommand<string> ExecuteCommand
        {
            get
            {
                if (_ExecuteCommand == null) _ExecuteCommand = new RelayCommand<string>(s =>
                {
                    if (cc.IsProcessRunning) cc.StopProcess();
                    cc.ClearOutput();
                    run("match", _Sources.Where(p => p.IsSelected).Select(p => p.Path).ToList());
                });
                return _ExecuteCommand;
            }
        }
        private RelayCommand<string> _StopCommand;
        public RelayCommand<string> StopCommand
        {
            get
            {
                if (_StopCommand == null) _StopCommand = new RelayCommand<string>(s =>
                {
                    if (cc.IsProcessRunning) cc.StopProcess();

                });
                return _StopCommand;
            }
        }

        private RelayCommand<ImageSource> _SelectCommand;
        public RelayCommand<ImageSource> SelectCommand
        {
            get
            {
                if (_SelectCommand == null) _SelectCommand = new RelayCommand<ImageSource>(s =>
                {
                    Selected = s.Path;
                    s.IsSelected = !s.IsSelected;
                });
                return _SelectCommand;
            }
        }
    }
}