using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GalaSoft.MvvmLight.CommandWpf;
using GalaSoft.MvvmLight;
namespace loader.ViewModel
{
    public class ImageSource : ViewModelBase
    {
        private string _Path;
        public string Path
        {
            get { return _Path; }
            set
            {
                _Path = value;
                RaisePropertyChanged("Path");
            }
        }
        private bool _IsSelected;
        public bool IsSelected
        {
            get { return _IsSelected; }
            set
            {
                _IsSelected = value;
                RaisePropertyChanged("IsSelected");
            }
        }
        public ImageSource(string p)
        {
            Path = p;
        }

    }
}
