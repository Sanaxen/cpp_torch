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

namespace DCGAN_Application
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        public static System.Drawing.Image CreateImage(string filename)
        {
            System.IO.FileStream fs = new System.IO.FileStream(
                filename,
                System.IO.FileMode.Open,
                System.IO.FileAccess.Read);
            System.Drawing.Image img = System.Drawing.Image.FromStream(fs);
            fs.Close();
            return img;
        }
        private void button1_Click(object sender, EventArgs e)
        {
            System.Random r = new System.Random(Environment.TickCount);
            textBox1.Text = r.Next(1000).ToString();
        }

        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var app = new System.Diagnostics.ProcessStartInfo();
            app.FileName = "dcgan_generate_test.exe";
            app.UseShellExecute = true;

            if (checkBox1.Checked) app.Arguments = " --gpu 1";
            else app.Arguments = " --gpu 0";
            app.Arguments += " --seed " + "\"" +textBox1.Text + "\"";
            decimal r = numericUpDown1.Value * numericUpDown1.Value;
            app.Arguments += " --batch" + " \"" + r.ToString() + "\"";
            if ( radioButton1.Checked)
            {
                System.IO.File.Copy("anime_model\\g_model.pt", "g_model.pt", true);
            }
            if (radioButton2.Checked)
            {
                System.IO.File.Copy("scenery_model\\g_model.pt", "g_model.pt", true);
            }
            if (radioButton3.Checked)
            {
                System.IO.File.Copy("beach_model\\g_model.pt", "g_model.pt", true);
            }
            if (radioButton4.Checked)
            {
                if (textBox2.Text == "" && openFileDialog1.ShowDialog() == DialogResult.OK)
                {
                    textBox2.Text = openFileDialog1.FileName;
                }
                System.IO.File.Copy(textBox2.Text, "g_model.pt", true);
            }


            string path = System.AppDomain.CurrentDomain.BaseDirectory;
            string cur = Directory.GetCurrentDirectory();

            Directory.SetCurrentDirectory(path);

            app.WindowStyle = System.Diagnostics.ProcessWindowStyle.Minimized;
            System.Diagnostics.Process p = System.Diagnostics.Process.Start(app);
            p.WaitForExit();

            System.IO.File.Copy(@".\generated_images\image_array99999.png", "image_array.png", true);
            pictureBox1.Image = CreateImage("image_array.png");
            Directory.SetCurrentDirectory(cur);
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void button3_Click(object sender, EventArgs e)
        {
            button1_Click(sender, e);
            button2_Click(sender, e);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            textBox2.Text = "";
        }
    }
}
