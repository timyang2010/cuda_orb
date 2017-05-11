using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Forms;
using System.Linq;
namespace loader.ViewModel
{
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
        /// <summary>
        /// Initializes a new instance of the MainViewModel class.
        /// </summary>
        public ObservableCollection<ImageSource> Sources { get { return _Sources; } }
        private ObservableCollection<ImageSource> _Sources = new ObservableCollection<ImageSource>();
        public MainViewModel()
        {
            
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
        Program _P = new Program();
        public Program P
        {
            get { return _P; }
        }
        private RelayCommand _LoadCommand; 
        public RelayCommand LoadCommand
        {
            get
            {
                if (_LoadCommand == null) _LoadCommand = new RelayCommand(() =>
                    {
                        OpenFileDialog ofd = new OpenFileDialog() { Multiselect = true };
                        if(ofd.ShowDialog()==DialogResult.OK)
                        {

                            foreach(var f in ofd.FileNames)
                            {
                                Sources.Add(new ImageSource(f));
                            }
                        }
                    });
                return _LoadCommand;
            }
        }
        private RelayCommand<string> _ExecuteCommand;
        public RelayCommand<string> ExecuteCommand
        {
            get
            {
                if (_ExecuteCommand == null) _ExecuteCommand = new RelayCommand<string>(s =>
                      {
                          P.run(s, Sources.Select(p => p.Path).ToList());
                      });
                return _ExecuteCommand;
            }
        }
        private RelayCommand<string> _SelectCommand;
        public RelayCommand<string> SelectCommand
        {
            get
            {
                if (_SelectCommand == null) _SelectCommand = new RelayCommand<string>(s =>
                {
                    Selected = s;
                });
                return _SelectCommand;
            }
        }
    }
}