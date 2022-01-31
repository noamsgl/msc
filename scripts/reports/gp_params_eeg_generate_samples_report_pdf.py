"""
Noam Siegel
30 Jan 2022
Generate EEG samples (interictal then ictal) and dump to PDF
"""
from decimal import Decimal

import gpytorch
import matplotlib.pyplot as plt
import torch
from borb.pdf.canvas.layout.image.chart import Chart
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from joblib import load
from matplotlib.figure import Figure

from msc import config
from msc.data_utils import get_time_as_str
from msc.models import EEGGPModel


def generate_sample_fig(params_generator, n_channels=8) -> Figure:
    """return a plt figure of a generated EEG sample"""
    # sample GP model params
    params, label = params_generator.sample()

    # build GP model from sampled params
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    times = torch.linspace(0, 1, 401)
    model = EEGGPModel([], [], likelihood)
    model.covar_module.raw_outputscale.data.fill_(params[0, 0])
    model.covar_module.base_kernel.raw_lengthscale.data.fill_(params[0, 1])

    # sample EEG from GP model
    sample = model(times).sample(torch.Size([n_channels]))

    # plot EEG sample
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_subplot()
    ax.set_xlabel("time (s)")
    for i in range(len(sample)):
        channel = sample[i]
        channel += i
        ax.plot(times, channel)
    return fig


if __name__ == '__main__':
    # initialize single column document
    doc = Document()
    page = Page()
    doc.append_page(page)
    layout = SingleColumnLayout(page)

    # add title
    layout.add(Paragraph(f"Generated interictal EEG", font_size=Decimal(24)))

    # Define number of samples
    N = 10

    # load params generator
    interictal_gmm_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\density_estimation_models\GMM_interictal_20220130T124223.joblib"
    interictal_gmm = load(interictal_gmm_fpath)

    # Generate interictals
    for i in range(N):
        layout.add(Paragraph(f"interictal_{i}"))
        layout.add(Chart(generate_sample_fig(interictal_gmm),
                         width=Decimal(450),
                         height=Decimal(256)
                         )
                   )

    # load params generator
    ictal_gmm_fpath = r"C:\Users\noam\Repositories\noamsgl\msc\results\density_estimation_models\GMM_ictal_20220130T215815.joblib"
    ictal_gmm = load(ictal_gmm_fpath)

    # Generate ictals
    for i in range(N):
        layout.add(Paragraph(f"ictal_{i}"))
        layout.add(Chart(generate_sample_fig(ictal_gmm),
                         width=Decimal(450),
                         height=Decimal(256)
                         )
                   )

    # dump to PDF
    with open(f"{config['PATH']['LOCAL']['RESULTS']}/reports/generated_samples_{get_time_as_str()}.pdf",
              "wb") as out_file_handle:
        PDF.dumps(out_file_handle, doc)
