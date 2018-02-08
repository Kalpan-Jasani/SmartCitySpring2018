import java.util.Random;
import java.util.Scanner;

/*
In this program, we simulate a pothole. 
The characteristics that can be set can be set in a file called characteristics.txt. Do not remove any line from that file. Just modify the values.

Also, some assumptions:
	The dimensions are 460 rows by 620 columns per row. So, I call each cell a pixel, and then give it some depth value.

	The data is given to standard output. But if you run ./runtest.sh (after changing its permissions using chmod), it does all the things and even starts the python script which uses this data. Also, it will correctly take input from the characteristics file
	
*/

class Snap
{
	static double[][] data = new double[460][620];	/*the underlying data that we will fill and then finally print out! */
	static int trackCount;	
	static Random r; 

	public static void main(String[] args)
	{
		int n1;		/*variables to store the indices to differntiate pothole pixels and other pixels */
		int n2;
		int n3;
		int m1;
		int m2;
		int m3;

		/*the next variables (6) are all fields that will be correctly initialized to their values as read from the file */
		final double roadLevel;
		final double deflectionPothole;
		final double deflectionRoad;
		final double lPercent;
		final double bPercent;
		final double potholeDepth;

		/*these next two variables store the actual deflection for each pizel, which is randomly generated */
		double deflectionInRoadActual;
		double deflectionInPotholeActual;
		
		String temp1, temp2;
		Scanner s;

		s = new Scanner(System.in);	/*the file is considered to be passed as standard input */
		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);	/*the substring will be from the index of '=' + 1, till the end of the string */
		lPercent = Double.parseDouble(temp2);

		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);
		bPercent = Double.parseDouble(temp2);
				
		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);
		potholeDepth = Double.parseDouble(temp2);

		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);
		roadLevel = Double.parseDouble(temp2);

		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);
		deflectionPothole = Double.parseDouble(temp2);

		temp1 = s.nextLine();
		temp2 = temp1.substring(temp1.indexOf('=') + 1);
		deflectionRoad = Double.parseDouble(temp2);	

		n2 = (int)((lPercent / 100.0) * 460);	/*the number of pixels per row for the pothole */
		n1 = (460 - n2) / 2;	/* the number of rows before pothole starts. Note that pothole is kept as a rectangle */
		n3 = 460 - n1 - n2;	/* the number of rows below after pothole finishes. */
		
		m2 = (int)((bPercent / 100.0) * 620);	/*number of columns in which pothole pixels are there. */
		m1 = (620 - m2) / 2;	/*the number of columns before pothole starts */
		m3 = 620 - m1 - m2;	/*the number of columns after pothole finishes */
		r = new Random();

	 	for(int i = 0; i < n1; i++)
		{
			for(int j = 0; j < 620; j++)
			{
				deflectionInRoadActual = (r.nextDouble() * 2 - 1) * deflectionRoad;
				arrayInsert(roadLevel + deflectionInRoadActual);
			}
		}

		for(int i = 0; i < n2; i++)
		{
			for(int j = 0; j < m1; j++)
			{
				deflectionInRoadActual = (r.nextDouble() * 2 - 1) * deflectionRoad;
				arrayInsert(roadLevel + deflectionInRoadActual);
			}
		
			for(int j = 0; j < m2; j++)
			{
				deflectionInPotholeActual = (r.nextDouble() * 2 - 1) * deflectionPothole;
				arrayInsert(potholeDepth + deflectionInPotholeActual); 	/*we insert a depth value plus a random small deflection initiated in the above line. */
			}
			for(int j = 0; j < m3; j++)
			{
				deflectionInRoadActual = (r.nextDouble() * 2 - 1) * deflectionRoad;
				arrayInsert(roadLevel + deflectionInRoadActual);
			}
		} 

		for(int i = 0; i < n3; i++)
		{
			for(int j = 0; j < 620; j++)
			{
				deflectionInRoadActual = (r.nextDouble() * 2 - 1) * deflectionRoad;
				arrayInsert(roadLevel + deflectionInRoadActual);
			}
		}
		
		for(int i = 0; i < 460; i++)	/* we print out the whole array in a specific format */
		{
			for(int j = 0; j < 619; j++)
			{
				System.out.print(data[i][j] + "\t");
			}
			System.out.println(data[i][619]);
		}
	}

/*
in this function, when it is called, a counter that stays in memory over function calls, is incremented. That is then used to determine where to put in the array, the argument that is received
Basically, arrayInsert can be a function that sequentially loads the data into the array !!!
 */

	static void arrayInsert(double numberValue)
	{
		int rowIndice;	/* the row indice for the value to be stored in the array */
		int columnIndice;	/* the column indice for the value to be stored in the array */

		/*
		we will calculate the appropriate indices based on arithmetic on the number provided
		*/
		rowIndice = trackCount / 620;	/* note that trackCount begins from 0. But then too, this case works. Otherwise, r= (trackCount - 1) / 620 */
		columnIndice = trackCount % 620;

		/*store the data into the array, as we now have finally got the row and column indice. */
		data[rowIndice][columnIndice] = numberValue;
		trackCount++;
	}
}
