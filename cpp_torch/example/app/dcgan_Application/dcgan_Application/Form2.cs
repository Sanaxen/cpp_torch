using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace dcgan_Application
{
    public partial class Training : Form
    {
        public Training()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var app = new System.Diagnostics.ProcessStartInfo();
            app.FileName = "dcgan.exe";
            app.UseShellExecute = true;

            if (checkBox1.Checked) app.Arguments = " --gpu 1";
            else app.Arguments = " --gpu 0";
            app.Arguments += " --data_root " + "\"" + textBox1.Text + "\"";
            app.Arguments += " --batch " + numericUpDown1.Value.ToString();
            app.Arguments += " --epoch " + numericUpDown2.Value.ToString();
            app.Arguments += " --lr " + textBox3.Text;
            app.Arguments += " --beta1 " + textBox4.Text;
            app.Arguments += " --drop_rate " + textBox5.Text;

            if (checkBox1.Checked)
            {
                app.Arguments += " --augment 1";
            }


            string path = System.AppDomain.CurrentDomain.BaseDirectory;
            string cur = System.IO.Directory.GetCurrentDirectory();

            System.IO.Directory.SetCurrentDirectory(path);

            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;
            System.Diagnostics.Process p = System.Diagnostics.Process.Start(app);
            p.WaitForExit();

            if (textBox2.Text != "")
            {
                System.IO.File.Copy("g_model.pt", textBox2.Text, true);
            }

            System.IO.Directory.SetCurrentDirectory(cur);
        }
    }
}
