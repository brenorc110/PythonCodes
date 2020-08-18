/*using System.IO;
using System;
using System.Collections.Generic;
using System.Drawing;
public class Shape
{
    // A few example members
    public int X { get; private set; }
    public int Y { get; private set; }
    public int Height { get; set; }
    public int Width { get; set; }

    // Virtual method
    public virtual void Draw()
    {
        Console.WriteLine("Performing base class drawing tasks");
    }
}

class Circle : Shape
{
    public override void Draw(this Graphics g, Pen pen,
                                  float centerX, float centerY, float radius)
    {
        g.DrawEllipse(pen, centerX - radius, centerY - radius,
                      radius + radius, radius + radius);
        Console.WriteLine("Drawing a circle");
        base.Draw();
    }
}

class Rectangle : Shape
{
    public override void Draw()
    {
        if (tempDraw != null)
                {
                    tempDraw = (Bitmap)snapshot.Clone();
                    Graphics g = Graphics.FromImage(tempDraw);
                    Pen myPen = new Pen(foreColor, lineWidth);
                    g.DrawRectangle(myPen, x1, y1, x2-x1, y2-y1);
                    myPen.Dispose();
                    e.Graphics.DrawImageUnscaled(tempDraw, 0, 0);
                    g.Dispose();
                }
        Console.WriteLine("Drawing a rectangle");
        base.Draw();
    }
}

class Triangle : Shape
{
    public override void Draw()
    {
        Console.WriteLine("Drawing a trangle");
        base.Draw();
    }
}

class Program
{
    static void Main(string[] args)
    {

        var shapes = new List<Shape>
        {
            new Rectangle(),
            new Triangle(),
            new Circle()
        };

        foreach (var shape in shapes)
        {
            shape.Draw();
        }

        // Keep the console open in debug mode.
        Console.WriteLine("Press any key to exit.");
        Console.ReadKey();
    }

}

/*
    Drawing a rectangle
    Performing base class drawing tasks
    Drawing a triangle
    Performing base class drawing tasks
    Drawing a circle
    Performing base class drawing tasks
*/