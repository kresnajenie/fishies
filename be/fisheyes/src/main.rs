extern crate hdf5;
extern crate ndarray;

use hdf5::{File, Result};
use ndarray::{ArrayD, Axis};

fn main() -> Result<()> {
    // Open the .h5ad file
    let file = File::open("../../data/4_week_full_labeled_celltype.h5ad")?;

    // Assuming the .obs dataset is stored as a named dataset and you're interested in "celltype"
    let dataset = file.dataset(".X")?;

    // Read the dataset data into an ndarray::ArrayD
    let data: ArrayD<f64> = dataset.read()?; // Adjust the type if necessary

    // Assuming "celltype" column is at a specific index, for example, 0
    let celltype_index = 0; // Replace with the actual index of "celltype"

    // Extract the "celltype" column. Note: This assumes data is 2D and "celltype" is numeric
    // For string data, you'd need a different approach since ndarray doesn't directly support strings
    let celltypes = data.index_axis(Axis(1), celltype_index);

    // Assuming you want to print or otherwise process `celltypes`
    println!("Celltypes: {:?}", celltypes);

    Ok(())
}
