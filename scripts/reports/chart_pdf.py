from decimal import Decimal

import matplotlib.pyplot as MatPlotLibPlot
import numpy as np
import pandas as pd
from borb.pdf.canvas.layout.image.chart import Chart
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from borb.pdf.canvas.layout.text.paragraph import Paragraph

from msc.dataset import UniformDataset


def create_plot(dataset, i) -> None:
    sample = dataset.get_train_y()[i][0]
    times = dataset.get_train_x()

    # plot
    fig = MatPlotLibPlot.figure()
    ax = fig.add_subplot()
    ax.plot(times, sample)
    # return
    return MatPlotLibPlot.gcf()


def main():
    doc: Document = Document()
    page: Page = Page()
    doc.append_page(page)

    layout: PageLayout = SingleColumnLayout(page)

    layout.add(Paragraph("Raw Data Plots of Uniform Dataset",font_size=Decimal(24)))
    # generate dataset
    dataset = UniformDataset(r"C:\Users\noam\Repositories\noamsgl\msc\results\epilepsiae\UNIFORM\20220106T165558")

    for i in range(10):
        layout.add(Chart(create_plot(dataset, i),
                         width=Decimal(400),
                         height=Decimal(256)
                         )
                   )

    with open("output.pdf", "wb") as out_file_handle:
        PDF.dumps(out_file_handle, doc)


if __name__ == "__main__":
    main()
