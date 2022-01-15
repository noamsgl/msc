from decimal import Decimal

import matplotlib.pyplot as plt
from borb.pdf.canvas.layout.image.chart import Chart
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from matplotlib.axes import Axes
from tqdm import tqdm

from msc import config
from msc.data_utils import get_time_as_str
from msc.dataset import UniformDataset, SeizuresDataset


def plot_sample(times, sample) -> None:
    # plot
    plt.clf()
    fig = plt.gcf()
    ax: Axes = fig.add_subplot()
    ax.plot(times, sample)
    ax.set_xlabel("time (s)")
    # return
    return plt.gcf()


def main(dataset_name="uniform"):
    dataset_name = dataset_name.lower()

    doc: Document = Document()
    page: Page = Page()
    doc.append_page(page)

    layout: PageLayout = SingleColumnLayout(page)

    layout.add(Paragraph("Raw Data Plots of Uniform Dataset", font_size=Decimal(24)))
    # load dataset
    if dataset_name == "uniform":
        dataset = UniformDataset(r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\UNIFORM\20220106T165558")
    elif dataset_name == "seizures":
        dataset = SeizuresDataset(r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\SEIZURES\20220103T101554")
    else:
        raise ValueError()

    times = dataset.get_train_x()
    samples = dataset.get_train_y(num_channels=10)
    for i in tqdm(range(len(samples))):
        for j in range(len(samples[i])):
            sample = samples[i][j]
            layout.add(Paragraph(f"Sample {i + 1}/{len(samples)}, Channel {j + 1}/{len(samples[i])}"))
            layout.add(Chart(plot_sample(times, sample),
                             width=Decimal(400),
                             height=Decimal(256)
                             )
                       )

    with open(f"{config['PATH']['LOCAL']['RESULTS']}/reports/{dataset_name.upper()}_{get_time_as_str()}.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, doc)


if __name__ == "__main__":
    main("seizures")
