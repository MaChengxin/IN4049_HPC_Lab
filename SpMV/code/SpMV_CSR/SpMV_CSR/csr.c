/*
CSR-Formation by Jianbing Jin, 2016-Jan-6
*/

#include "csr.h"

int main(int argc, char **argv)
{
	Read_Matrix();
	CSR_Transformation();
	Write_Matrix();
	Clean_Up();
	return 0;
}

/*
Read_Matrix() reads full-matrix-input2.dat and
stores the row index, column index, and the value into
*row, *col, and *val_in.
It also stores the size of the matrix and the number of non-zero elements.
*/
void Read_Matrix()
{
	Debug("Reading matrix info...", 0);

	int s;
	FILE *f;
	int row_idx;
	int col_idx;
	float val;

	f = fopen("input/test_mat.dat", "r");
	if (f == NULL)
		Debug("Error opening test_mat.dat", 1);

	fscanf(f, "%i %i %i\n", &row_size, &col_size, &none_zero_num);

	if ((row = malloc(none_zero_num * sizeof(*row))) == NULL)
		Debug("Read_Matrix : malloc(row) failed", 1);
	if ((col = malloc(none_zero_num * sizeof(*col))) == NULL)
		Debug("Read_Matrix : malloc(col) failed", 1);
	if ((val_in = malloc(none_zero_num * sizeof(*val_in))) == NULL)
		Debug("Read_Matrix : malloc(val_in) failed", 1);

	for (int i = 0; i < none_zero_num; i++)
	{
		row[i] = 0;
		col[i] = 0;
		val_in[i] = 0.0;
	}

	for (int i = 0; i < none_zero_num; i++)
	{
		s = fscanf(f, "%i %i %f\n", &row_idx, &col_idx, &val);
		if (s == 3)
		{
			row[i] = row_idx;
			col[i] = col_idx;
			val_in[i] = val;
		}
	}

	fclose(f);
	Debug("Matrix info read.", 0);
}

void CSR_Transformation()
{
	Debug("Generating matrix representation with CSR format...", 0);
	int* each_row_counter = (int*)malloc((row_size + 1) * sizeof(int));
	int* Matrix_row_counter = (int*)malloc((row_size + 1) * sizeof(int));
	int i, j;
	int CSR_counter = 0;

	for (i = 0; i < row_size + 1; i++)
	{
		each_row_counter[i] = 0;
		Matrix_row_counter[i] = 0;
	}

	for (i = 0; i < none_zero_num; i++)
	{
		j = row[i];
		each_row_counter[j] += 1;
	}

	int** col_Matrix = (int **)malloc((row_size + 1) * sizeof(int *));
	if (col_Matrix == NULL)
		Debug("CSR_Transformation : malloc (col_1) failed", 1);

	float** val_Matrix = (float **)malloc((row_size + 1) * sizeof(float *));
	if (val_Matrix == NULL)
		Debug("CSR_Transformation : malloc (val_1) failed", 1);

	for (i = 0; i < row_size + 1; i++)
	{
		if ((col_Matrix[i] = malloc((each_row_counter[i]) * sizeof(int))) == NULL)
			Debug("CSR_Transformation : malloc (col_2) failed", 1);
		if ((val_Matrix[i] = malloc((each_row_counter[i]) * sizeof(float))) == NULL)
			Debug("CSR_Transformation : malloc (val_2) failed", 1);
	}

	for (i = 0; i < row_size + 1; i++)
	{
		for (j = 1; j < each_row_counter[i]; j++)
		{
			val_Matrix[i][j] = 0.0;
			col_Matrix[i][j] = 0;
		}
	}

	for (i = 0; i < none_zero_num; i++)
	{
		j = row[i];
		Matrix_row_counter[j]++;
		val_Matrix[j][Matrix_row_counter[j]] = val_in[i];
		col_Matrix[j][Matrix_row_counter[j]] = col[i];
	}

	if ((row_ptr = malloc((row_size + 1) * sizeof(*row_ptr))) == NULL)
		Debug("Read_Matrix : malloc(row_ptr) failed", 1);
	if ((col_idx = malloc(none_zero_num * sizeof(*col_idx))) == NULL)
		Debug("Read_Matrix : malloc(col_idx) failed", 1);
	if ((val_out = malloc(none_zero_num * sizeof(*val_out))) == NULL)
		Debug("Read_Matrix : malloc(val_out) failed", 1);

	for (i = 0; i < row_size + 1; i++)
		row_ptr[i] = 0;

	for (i = 0; i < none_zero_num; i++)
	{
		col_idx[i] = 0;
		val_out[i] = 0;
	}

	for (i = 1; i < row_size + 1; i++)
	{
		row_ptr[i] = each_row_counter[i] + row_ptr[i - 1];
	}

	for (i = 1; i < row_size + 1; i++)
	{
		for (j = 1; j < each_row_counter[i] + 1; j++)
		{
			val_out[CSR_counter] = val_Matrix[i][j];
			col_idx[CSR_counter] = col_Matrix[i][j];
			CSR_counter++;
		}
	}

	free(each_row_counter);
	free(Matrix_row_counter);
	free(col_Matrix);
	free(val_Matrix);
	Debug("Matrix representation with CSR format generated.", 0);
}

/*
Write_Matrix() generates the file for the CUDA application.
*/
void Write_Matrix()
{
	Debug("Writing transformed matrix info...", 0);

	FILE *f;

	if ((f = fopen("csr_matrix.dat", "w")) == NULL)
		Debug("Write_Matrix: fopen failed", 1);

	fprintf(f, "(row*column, none_zero_num): (%i*%i, %i)\n", row_size, col_size, none_zero_num);

	fprintf(f, "row_ptr:\n");
	for (int i = 0; i < row_size + 1; i++)
		fprintf(f, "%i\n", row_ptr[i]);

	fprintf(f, "col_idx and val:\n");
	for (int i = 0; i < none_zero_num; i++)
		fprintf(f, "%i\t %f\n", col_idx[i] - 1, val_out[i]);

	fclose(f);

	Debug("Transformed matrix info written.", 0);
}

void Clean_Up()
{
	Debug("Cleaning up...", 0);

	free(row);
	free(col);
	free(val_in);

	Debug("Cleaned up.", 0);
}

void Debug(char *mesg, int terminate)
{
	if (DEBUG || terminate)
		printf("%s\n", mesg);
	if (terminate)
		exit(1);
}
