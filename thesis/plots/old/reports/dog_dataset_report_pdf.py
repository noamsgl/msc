"""
Noam Siegel
30 Jan 2022
Generate a PDF report with EEG samples
"""
from decimal import Decimal
from itertools import count

import matplotlib.pyplot as plt
from borb.pdf.canvas.layout.image.chart import Chart
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from msc import config
from msc.data_utils import get_time_as_str
from msc.dataset import DogDataset


def plot_sample(group) -> Figure:
    # plot EEG sample
    plt.clf()
    fig = plt.gcf()
    ax: Axes = fig.add_subplot()
    ax.set_xlabel("time (s)")
    data_cols = [col for col in group.columns if 'Ecog' in col]
    assert len(data_cols) > 0, "Error: data cols not found"
    for i in range(len(data_cols)):
        channel = group[data_cols[i]]
        channel += i
        ax.plot(group.time, channel)
    # return
    return plt.gcf()


def main(dataset_dir: str, dataset_name):
    # get dataset name
    dataset_name = dataset_name.lower()
    # initialize PDF document
    doc: Document = Document()
    page: Page = Page()
    doc.append_page(page)

    layout: PageLayout = SingleColumnLayout(page)

    # add Title Heading
    layout.add(Paragraph(f"Raw Data Plots of {dataset_name.capitalize()} Dataset", font_size=Decimal(24)))

    # load dataset
    dataset = DogDataset(dataset_dir)
    counter = count()
    samples_df = dataset.normalized_samples()
    for name, group in samples_df.groupby('fname'):
        count_id = next(counter)
        # if count_id > 4:
        #     break
        layout.add(Paragraph(f"{dataset_name.capitalize()} sample {name}"))
        layout.add(Chart(plot_sample(group),
                         width=Decimal(450),
                         height=Decimal(256)
                         )
                   )

    with open(f"{config['PATH']['LOCAL']['RESULTS']}/reports/{dataset_name.upper()}_{get_time_as_str()}.pdf",
              "wb") as out_file_handle:
        PDF.dumps(out_file_handle, doc)


if __name__ == "__main__":
    dataset_dir = r"C:\Users\noam\Repositories\noamsgl\msc\data\seizure-detection\Dog_1"
    main(dataset_dir, dataset_name="Dog_1")
