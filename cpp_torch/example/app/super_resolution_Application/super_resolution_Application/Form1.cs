using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace super_resolution_Application
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            string[] cmds = System.Environment.GetCommandLineArgs();
            if ( cmds.Length >= 2)
            {
                openFileDialog1.FileName = cmds[1];
                pictureBox1.Image = System.Drawing.Image.FromFile(cmds[1]);
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            DialogResult res = openFileDialog1.ShowDialog();
        }

        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
            pictureBox1.Image = System.Drawing.Image.FromFile(openFileDialog1.FileName);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var app = new System.Diagnostics.ProcessStartInfo();
            app.FileName = "super_resolution_espcn_test.exe";
            app.UseShellExecute = true;
            app.Arguments = "\"" + openFileDialog1.FileName + "\"";

            string directoryName = System.IO.Path.GetDirectoryName(openFileDialog1.FileName);
            string fileName = System.IO.Path.GetFileNameWithoutExtension(openFileDialog1.FileName);
            string extension = System.IO.Path.GetExtension(openFileDialog1.FileName);

            string path = System.AppDomain.CurrentDomain.BaseDirectory;
            string cur = Directory.GetCurrentDirectory();

            Directory.SetCurrentDirectory(path);

            System.Diagnostics.Process p = System.Diagnostics.Process.Start(app);
            p.WaitForExit();

            string outfile = directoryName + "\\" + fileName + "_super_res" + extension;
            System.IO.File.Copy("super_resolution_test_999.png", outfile, true);

            Directory.SetCurrentDirectory(cur);


            Directory.SetCurrentDirectory(path);
            var app2 = new System.Diagnostics.ProcessStartInfo();
            app2.FileName = "super_resolution_Application.exe";
            app2.Arguments = "\"" + outfile + "\"";
            app2.UseShellExecute = true;
            System.Diagnostics.Process.Start(app2);
        }
    }
}
