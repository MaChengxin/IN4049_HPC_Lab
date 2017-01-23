/*
CSR5-Formation
By Jianbing Jin   2016-Jan-6
*/

#include "csr5.h"

int main(int argc, char **argv)
{
	Read_Matrix();

	CSR_Transformation();

	Compute_Sigma();

	CSR5_Transformation();

	Write_Matrix_Info();

	Clean_Up();

	return 0;
}

void Read_Matrix()
{
	int x, s, i = 0;
	FILE *f;
	int source_x;
	int source_y;
	float source_val;
	Debug("Read_Matrix", 0);

	f = fopen("input/test_mat.dat", "r");
	if (f == NULL)
		Debug("Error opening test_mat.dat", 1);

	// read the row_num column_num and none_zero_num of the input matrix
	fscanf(f, "%i %i %i\n", &row_num, &col_num, &none_zero_num);
	printf("the fisrt row is %i %i %i\n", row_num, col_num, none_zero_num);
	/* allocate memory for the input matrix */
	if ((row = malloc(none_zero_num * sizeof(*row))) == NULL)
		Debug("Read_Matrix : malloc(row) failed", 1);
	if ((col = malloc(none_zero_num * sizeof(*col))) == NULL)
		Debug("Read_Matrix : malloc(col) failed", 1);
	if ((val_input = malloc(none_zero_num * sizeof(*val_input))) == NULL)
		Debug("Read_Matrix : malloc(val_input) failed", 1);

	/* set all values to '0' */
	for (x = 0; x < none_zero_num; x++)
	{
		row[x] = 0;
		col[x] = 0;
		val_input[x] = 0.0;
	}

	/* put sources in field */
	for (x = 0; x < none_zero_num; x++)
	{
		s = fscanf(f, "%i %i %f\n", &source_x, &source_y, &source_val);
		if (s == 3)
		{
			row[x] = source_x;
			col[x] = source_y;
			val_input[x] = source_val;
		}
	}
	fclose(f);

	/* for(x = 0; x < none_zero_num; x++)
	 {
	   printf("the %i th is: %i, %i, %f\n", x, row[x], col[x], val_input[x]);
	 }*/
}

void CSR_Transformation()
{
	int* each_row_counter = (int*)malloc((row_num + 1) * sizeof(int));
	int* Matrix_row_counter = (int*)malloc((row_num + 1) * sizeof(int));
	int counter = 0;
	int i, j;
	int each_val_col;
	int CSR_counter = 0;

	for (i = 0; i < row_num + 1; i++)
	{
		each_row_counter[i] = 0;
		Matrix_row_counter[i] = 0;
	}

	for (i = 0; i < none_zero_num; i++)     //count the non_zero value in each row
	{
		j = row[i];
		each_row_counter[j] = each_row_counter[j] + 1;
	}

	if ((row_Matrix = malloc((row_num + 1) * sizeof(*row_Matrix))) == NULL)
		Debug("CSR_Transformation : malloc (row_1) failed", 1);
	if ((col_Matrix = malloc((row_num + 1) * sizeof(*col_Matrix))) == NULL)
		Debug("CSR_Transformation : malloc (col_1) failed", 1);
	if ((val_Matrix = malloc((row_num + 1) * sizeof(*val_Matrix))) == NULL)
		Debug("CSR_Transformation : malloc (val_1) failed", 1);
	for (i = 0; i < row_num + 1; i++)
	{
		if ((row_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**row_Matrix))) == NULL)
			Debug("CSR_Transformation : malloc (row_2) failed", 1);
		if ((col_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**col_Matrix))) == NULL)
			Debug("CSR_Transformation : malloc (col) failed", 1);
		if ((val_Matrix[i] = malloc((each_row_counter[i]) * sizeof(**val_Matrix))) == NULL)
			Debug("CSR_Transformation : malloc (val_2) failed", 1);
	}

	for (i = 1; i < row_num; i++)
	{
		for (j = 1; j < each_row_counter[i]; j++)
		{
			val_Matrix[i][j] = 0.0;
			row_Matrix[i][j] = 0;  // none sense, or = i;
			col_Matrix[i][j] = 0;
		}
	}

	for (i = 0; i < none_zero_num; i++)
	{
		j = row[i];
		Matrix_row_counter[j]++;
		val_Matrix[j][Matrix_row_counter[j]] = val_input[i];
		row_Matrix[j][Matrix_row_counter[j]] = row[i];
		col_Matrix[j][Matrix_row_counter[j]] = col[i];
	}
	printf("!^_^! ^_^! ^_^! ^_^! ^_^!\n");

	if ((row_ptr = malloc((row_num + 1) * sizeof(*row_ptr))) == NULL)
		Debug("Read_Matrix : malloc(row) failed", 1);
	if ((col_idx = malloc(none_zero_num * sizeof(*col_idx))) == NULL)
		Debug("Read_Matrix : malloc(col) failed", 1);
	if ((val_output = malloc(none_zero_num * sizeof(*val_output))) == NULL)
		Debug("Read_Matrix : malloc(val_input) failed", 1);

	for (i = 0; i < row_num + 1; i++)
		row_ptr[i] = 0;
	//set the row_ptr, col_idx and val_output 0
	for (i = 0; i < none_zero_num; i++)
	{
		col_idx[i] = 0;
		val_output[i] = 0;
	}


	for (i = 1; i < row_num + 1; i++)
	{
		row_ptr[i] = each_row_counter[i] + row_ptr[i - 1];
	}

	for (i = 1; i < row_num + 1; i++)
	{
		for (j = 1; j < each_row_counter[i] + 1; j++)
		{
			val_output[CSR_counter] = val_Matrix[i][j];
			col_idx[CSR_counter] = col_Matrix[i][j] - 1;  // the fortran is different from c language for the strage of matrix
			CSR_counter++;
		}
	}
}

void Compute_Sigma()
{
	int r = 4;
	int s = 32;
	int t = 256;
	int u = 4;
	int division_result;

	division_result = (int)(none_zero_num / row_num);
	if (division_result < r)
		sigma = r;
	else if (division_result <= s)
		sigma = division_result;
	else if (division_result <= t)
		sigma = s;
	else
		sigma = u;

	printf("sigma is %i\n", sigma);
}

void CSR5_Transformation()
{
	Compute_Tile_Ptr();
	Compute_Tile_Val();
	Compute_Tile_Desc();
}

void Write_Matrix_Info()
{
	int x, y;
	FILE *f;

	if ((f = fopen("csr_matrix.dat", "w")) == NULL)
		Debug("Write_Matrix_Info : fopen failed", 1);

	Debug("Write_Matrix_Info", 0);
	fprintf(f, "csr_matrix_formate(row*column, none_zero_num): %i*%i, %i\n", row_num, col_num, none_zero_num);
	fprintf(f, "row_ptr:\n");
	for (x = 0; x < row_num + 1; x++)
	{
		fprintf(f, "%i\n", row_ptr[x]);
	}

	fprintf(f, "col_idx and val:\n");

	for (x = 0; x < none_zero_num; x++)
		fprintf(f, "%i %f\n", col_idx[x], val_output[x]);

	fclose(f);
}

void Clean_Up()
{
	Debug("Clean_Up", 0);

	free(row);
	free(col);
	free(val_input);
	free(row_Matrix);
	free(val_Matrix);
	free(col_Matrix);
	free(val_output);
	free(col_idx);
	free(row_ptr);
}

void Compute_Tile_Ptr()
{
	int tid, bnd;

	if ((none_zero_num % (omega * sigma)) > 0)
		p = (none_zero_num / (omega * sigma)) + 1;
	else
		p = none_zero_num / (omega * sigma);

	if ((tile_ptr = (int *)malloc((p + 1) * sizeof(int))) == NULL)
		Debug("Compute_Tile_Ptr : malloc(tile_ptr) failed", 1);
	memset(tile_ptr, 0, (p + 1) * sizeof(int));
	// for (int i = 0; i < p + 1; i++) printf("tile_ptr[%d] is %d\n", i, tile_ptr[i]);

	for (tid = 0; tid < p; tid++)
	{
		bnd = tid * omega * sigma;
		for (int i = 0; i < row_num; i++)
		{
			if (bnd < row_ptr[i + 1] && bnd >= row_ptr[i])
				tile_ptr[tid] = i;
		}
	}

	// TODO: make sure why the last element in tile_ptr is the row size
	tile_ptr[p] = row_num;

	for (tid = 0; tid < p + 1; tid++)
		printf("tile_ptr[%i] is %i\n", tid, tile_ptr[tid]);

	for (tid = 0; tid < p; tid++)  //check whether there is a empty row in the tile[i]
	{
		for (int rid = tile_ptr[tid]; rid < tile_ptr[tid + 1] + 1; rid++)
		{
			if (row_ptr[rid] == row_ptr[rid + 1])
			{
				tile_ptr[tid] = -tile_ptr[tid];
				break;
			}
		}
	}
}

void Compute_Tile_Val()
{
	int num_mat_squre = none_zero_num / (omega * sigma);

	if ((tile_val_in_mat = malloc(num_mat_squre * sizeof(*tile_val_in_mat))) == NULL)
		Debug("CSR5_Transformation_tile_val_in_mat : malloc (tile_val_in_mat) failed", 1);
	if ((tile_col_idx_in_mat = malloc(num_mat_squre * sizeof(*tile_col_idx_in_mat))) == NULL)
		Debug("CSR5_Transformation_tile_col_idx_in_mat : malloc (tile_col_idx_in_mat) failed", 1);

	for (int i = 0; i < num_mat_squre; i++)
	{
		if ((tile_val_in_mat[i] = malloc((sigma) * sizeof(**tile_val_in_mat))) == NULL)
			Debug("CSR5_Transformation_tile_val_in_mat: malloc (tile_val_in_mat[i]) failed", 1);
		if ((tile_col_idx_in_mat[i] = malloc((sigma) * sizeof(**tile_col_idx_in_mat))) == NULL)
			Debug("CSR5_Transformation_tile_col_idx_in_mat: malloc (tile_col_idx_in_mat[i]) failed", 1);

		for (int j = 0; j < sigma; j++)
		{
			if ((tile_val_in_mat[i][j] = malloc((omega) * sizeof(***tile_val_in_mat))) == NULL)
				Debug("CSR5_Transformation_tile_val_in_mat : malloc (tile_val_in_mat[i][j]) failed", 1);
			if ((tile_col_idx_in_mat[i][j] = malloc((omega) * sizeof(***tile_col_idx_in_mat))) == NULL)
				Debug("CSR5_Transformation_tile_col_idx_in_mat : malloc (tile_col_idx_in_mat[i][j]) failed", 1);
		}
	}

	for (int count = 0; count < num_mat_squre * sigma * omega; count++)
	{
		int i = count / (sigma * omega);  // the tile index
		int j = (count % (sigma * omega)) % sigma;  //the row index
		int k = (count % (sigma * omega)) / sigma;  //the column index

		tile_val_in_mat[i][j][k] = val_output[count];
		tile_col_idx_in_mat[i][j][k] = col_idx[count];
	}

	for (int i = 0; i < num_mat_squre; i++)
	{
		for (int j = 0; j < sigma; j++)
		{
			for (int k = 0; k < omega; k++)
			{
				// printf("tile_val_in_mat[%i][%i][%i] is %f\n", i, j, k, tile_val_in_mat[i][j][k]);
				printf("tile_col_idx_in_mat[%i][%i][%i] is %i\n", i, j, k,  tile_col_idx_in_mat[i][j][k]);
			}
		}
	}

	int num_rest_element = none_zero_num % (sigma*omega);

	if (num_rest_element != 0)
	{
		int *tile_val_in_array = (float*)malloc(num_rest_element * sizeof(float));
		if (tile_val_in_array == NULL)
			Debug("malloc tile_val_in_array failed.\n", 1);

		int *tile_col_idx_in_array = (int*)malloc(num_rest_element * sizeof(int));
		if (tile_col_idx_in_array == NULL)
			Debug("malloc tile_col_idx_in_array failed.\n", 1);

		for (int count = 0; count < num_rest_element; count++)
		{
			tile_val_in_array[count] = val_output[num_mat_squre * sigma * omega + count];
			tile_col_idx_in_array[count] = col_idx[num_mat_squre * sigma * omega + count];
			// printf("tile_val_in_array[%d] is %f\n", count, tile_val_in_array[count]);
			printf("tile_col_idx_in_array[%d] is %d\n", count, tile_col_idx_in_array[count]);
		}
	}
}

void Compute_Tile_Desc()
{
	int i, j, m;
	int count;
	//allocate memory for the bit_flag[][]
	if ((tile_bit_flag = malloc((p - 1) * sizeof(*tile_bit_flag))) == NULL)    // setup p-1 tiles
		Debug("CSR5_Transformation_tile_bit_flag : malloc (tile_bit_flag) failed", 1);
	for (i = 0; i < (p - 1); i++)
	{
		if ((tile_bit_flag[i] = malloc((sigma) * sizeof(**tile_bit_flag))) == NULL)  // every tile has sigma row
			Debug("CSR5_Transformation_tile_bit_flag: malloc (tile_bit_flag) failed", 1);
		for (j = 0; j < sigma; j++)
		{
			if ((tile_bit_flag[i][j] = malloc((omega) * sizeof(***tile_bit_flag))) == NULL) // every row in the  tile has w column
				Debug("CSR5_Transformation_tile_bit_flag: malloc (tile_bit_flag) failed", 1);
		}
	}

	for (count = 0; (count < (p - 1) * sigma * omega); count++)
	{
		// computer the corresponding index i j m for each count;
		i = count / (sigma * omega);  // tile index
		j = (count % (sigma * omega)) / sigma; //
		m = (count % (sigma * omega)) % sigma; //
		tile_bit_flag[i][m][j] = False;  //as the tile 
	}

	for (count = 0; count < row_num; count++)
	{
		i = row_ptr[count] / (sigma * omega);
		j = (row_ptr[count] % (sigma * omega)) / sigma;
		m = (row_ptr[count] % (sigma * omega)) % sigma;

		tile_bit_flag[i][m][j] = True;
	}

	for (i = 0; i < (p - 1); i++)
	{
		tile_bit_flag[i][0][0] = True;
	}

	for (i = 0; i < (p - 1); i++)
	{
		for (j = 0; j < sigma; j++)
		{
			for (m = 0; m < omega; m++)
			{
				printf("tile_bit_flag[%i][%i][%i] is %i\n", i, m, j, tile_bit_flag[i][m][j]);
			}
		}
	}
	///allocate memory for the y_offset and seg_offset
	if ((tile_y_offset = malloc((p - 1) * sizeof(*tile_y_offset))) == NULL)
		Debug("CSR5_Transformation_tile_y_offset : malloc (tile_y_offset) failed", 1);
	if ((tile_seg_offset = malloc((p - 1) * sizeof(*tile_y_offset))) == NULL)
		Debug("CSR5_Transformation_tile_seg_offset : malloc (tile_seg_offset) failed", 1);
	for (i = 0; i < p - 1; i++)
	{
		if ((tile_y_offset[i] = malloc((omega) * sizeof(**tile_y_offset))) == NULL)
			Debug("CSR5_Transformation_tile_y_offset: malloc (tile_y_offset) failed", 1);
		if ((tile_seg_offset[i] = malloc((omega) * sizeof(*tile_seg_offset))) == NULL)
			Debug("CSR5_Transformation_tile_seg_offset : malloc (tile_seg_offset) failed", 1);
	}
	//compute the y_offset//
	for (i = 0; i < (p - 1); i++)
	{
		int bais = 0;
		for (m = 0; m < omega; m++)
		{
			tile_y_offset[i][m] = bais;
			for (j = 0; j < sigma; j++)
			{
				bais = bais + tile_bit_flag[i][j][m];
			}
			printf("tile_y_offset[%i][%i] is %i\n", i, m, tile_y_offset[i][m]);
		}

		//compute the seg_offset//
		for (m = 0; m < omega; m++)
		{
			int seg = 0;
			for (j = 0; j < sigma; j++)
			{
				seg = max(tile_bit_flag[i][j][m], seg);
			}
			tile_seg_offset[i][m] = 1 - seg;
		}

		for (m = 0; m < omega - 1; m++)
		{
			tile_seg_offset[i][m] = tile_seg_offset[i][m + 1];
		}
		tile_seg_offset[i][m] = 0;
		for (m = 0; m < omega; m++)
		{
			printf("tile_seg_offset[%i][%i] is %i\n", i, m, tile_seg_offset[i][m]);
		}
	}

	//allocate memory for the empty_offset
	if ((tile_empty_offset = malloc((p - 1) * sizeof(*tile_empty_offset))) == NULL)
		Debug("CSR5_Transformation_tile_empty_offset : malloc (tile_empty_offset) failed", 1);

	// int empty_tile_true_counter[p - 1];
	int* empty_tile_true_counter = (int*)malloc((p - 1) * sizeof(int));  //calculate the number of "True" in each tile with empty row!!!

	for (i = 0; i < (p - 1); i++)
	{
		empty_tile_true_counter[i] = 0;
		if (tile_ptr[i] <= 0)
			for (m = 0; m < omega; m++)
			{
				for (j = 0; j < sigma; j++)
				{
					if (tile_bit_flag[i][j][m] == True)
						empty_tile_true_counter[i]++;
				}
			}
		printf("empty_tile_true_couter[%i] is %i\n", i, empty_tile_true_counter[i]);
	}

	for (i = 0; i < (p - 1); i++)
	{                            //empty_tile_true_counter[i]
		if ((tile_empty_offset[i] = malloc(empty_tile_true_counter[i] * sizeof(**tile_empty_offset))) == NULL)
			Debug("CSR5_Transformation_tile_empty_offset: malloc (tile_empty_offset) failed", 1);
		empty_tile_true_counter[i] = 0; // it will be used in the next program
	}

	for (i = 0; i < (p - 1); i++)
	{
		int row_ptr_counter = 0;
		int tid;
		if (tile_ptr[i] <= 0)
			for (m = 0; m < omega; m++)
			{
				for (j = 0; j < sigma; j++)
				{
					if (tile_bit_flag[i][j][m] == True)
					{
						tid = i * sigma * omega + m * omega + j;
						for (row_ptr_counter = 0; row_ptr_counter < row_num; row_ptr_counter++)
						{
							if ((tid >= row_ptr[row_ptr_counter]) && (tid <= row_ptr[row_ptr_counter + 1]))
							{
								printf("hehehehehehhehehhe\n");
								tile_empty_offset[i][empty_tile_true_counter[i]] = row_ptr_counter + tile_ptr[i]; // equal to tile_empty[][] = row_ptr_counter - remove_sign(tile_ptr[i])
																												  // printf("tile_empty_offset[%i][%i] is %i\n", i, empty_tile_true_counter[i], tile_empty_offset[i][empty_tile_true_counter[i]]);
							}
						}
						empty_tile_true_counter[i]++;
					}
				}
			}
	}

	for (i = 0; i < (p - 1); i++)
	{
		if (tile_ptr[i] <= 0)
		{
			for (j = 0; j < empty_tile_true_counter[i]; j++)
			{
				printf("tile_empty_offset[%i][%i] is %i\n", i, j, tile_empty_offset[i][j]);
			}
		}
	}
}

void Debug(char *mesg, int terminate)
{
	if (DEBUG || terminate)
		printf("%s\n", mesg);
	if (terminate)
		exit(1);
}
