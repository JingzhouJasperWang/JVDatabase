const chartGroups = [
  {
    title: "Database charts",
    intro: "Core descriptive views from the JV database and the pic1-pic6 analysis set.",
    charts: [
      {
        title: "JV events by broad sector",
        caption: "Automotive, Energy, and Aerospace / Defense are the largest event groups.",
        file: "results/graphs/pic1.png"
      },
      {
        title: "JV events by announcement year",
        caption: "Recent years carry most of the observed activity; 2025 is the peak year in this database.",
        file: "results/graphs/pic2.png"
      },
      {
        title: "Partner-role buckets",
        caption: "Operating and shared-active roles dominate partner rows; passive/financial is a small class.",
        file: "results/graphs/pic3.png"
      },
      {
        title: "Ownership structure mix",
        caption: "Equal or 50:50 ownership is the most common coded structure.",
        file: "results/graphs/pic4.png"
      },
      {
        title: "Passive/financial events by sector",
        caption: "Digital infrastructure and infrastructure/logistics account for most passive/financial events.",
        file: "results/graphs/pic5.png"
      },
      {
        title: "Cross-border mix",
        caption: "The database is heavily cross-border, with 393 of 460 events crossing jurisdictions.",
        file: "results/graphs/pic6.png"
      }
    ]
  },
  {
    title: "Model overview",
    intro: "Summary charts for model performance, class balance, and selected model behavior.",
    charts: [
      {
        title: "Overall best model summary",
        caption: "A compact comparison of positive rates and best-model metrics across event and partner levels.",
        file: "results/graphs/overall_best_model_summary.png"
      },
      {
        title: "Event-level class balance",
        caption: "The event-level training set has a 10.5% positive label rate.",
        file: "results/graphs/event_level_class_balance.png"
      },
      {
        title: "Partner-level class balance",
        caption: "The partner-level training set has a 4.6% positive label rate.",
        file: "results/graphs/partner_level_class_balance.png"
      },
      {
        title: "Event-level holdout metrics",
        caption: "Balanced logistic regression is strongest by holdout average precision.",
        file: "results/graphs/event_level_holdout_model_comparison.png"
      },
      {
        title: "Partner-level holdout metrics",
        caption: "Balanced random forest performs best on the partner-level holdout split.",
        file: "results/graphs/partner_level_holdout_model_comparison.png"
      }
    ]
  },
  {
    title: "Cross-validation and curves",
    intro: "Model comparison charts and threshold-free classification curves.",
    charts: [
      {
        title: "Event-level cross-validation comparison",
        caption: "Cross-validation favors the balanced logistic regression event model.",
        file: "results/graphs/event_level_cv_model_comparison.png"
      },
      {
        title: "Partner-level cross-validation comparison",
        caption: "Cross-validation shows useful but weaker signal at partner level.",
        file: "results/graphs/partner_level_cv_model_comparison.png"
      },
      {
        title: "Event-level precision-recall curve",
        caption: "Precision-recall is the key view because the positive class is rare.",
        file: "results/graphs/event_level_precision_recall_curve.png"
      },
      {
        title: "Partner-level precision-recall curve",
        caption: "Partner-level prediction is harder because positives are only 4.6% of rows.",
        file: "results/graphs/partner_level_precision_recall_curve.png"
      },
      {
        title: "Event-level ROC curve",
        caption: "The event model separates positive and negative examples well in ROC space.",
        file: "results/graphs/event_level_roc_curve.png"
      },
      {
        title: "Partner-level ROC curve",
        caption: "Partner-level ROC looks strong, but precision-recall is more conservative.",
        file: "results/graphs/partner_level_roc_curve.png"
      }
    ]
  },
  {
    title: "Best model diagnostics and feature importance",
    intro: "Confusion matrices and permutation importance for the selected models.",
    charts: [
      {
        title: "Event-level confusion matrix",
        caption: "At the selected threshold, the event model finds 8 of 12 positives in the holdout set.",
        file: "results/graphs/event_level_best_model_confusion_matrix.png"
      },
      {
        title: "Partner-level confusion matrix",
        caption: "At the selected threshold, the partner model finds 5 of 12 positives in the holdout set.",
        file: "results/graphs/partner_level_best_model_confusion_matrix.png"
      },
      {
        title: "Event-level feature importance",
        caption: "Sector is the largest event-level predictor by average precision drop.",
        file: "results/graphs/event_level_feature_importance.png"
      },
      {
        title: "Partner-level feature importance",
        caption: "Sector, ownership percentage, country/region, and year carry the main partner-level signal.",
        file: "results/graphs/partner_level_feature_importance.png"
      }
    ]
  }
];

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function renderCharts() {
  const root = document.querySelector("#chart-groups");
  if (!root) return;

  chartGroups.forEach((group) => {
    const section = createElement("section", "chart-group");
    const header = createElement("div", "chart-group-header");
    const headingWrap = document.createElement("div");
    const heading = createElement("h3", null, group.title);
    const intro = createElement("p", null, group.intro);
    headingWrap.append(heading, intro);
    header.append(headingWrap);

    const grid = createElement("div", "chart-grid");
    group.charts.forEach((chart) => {
      const card = createElement("article", "chart-card");
      const frame = createElement("div", "chart-frame");
      const img = document.createElement("img");
      img.src = chart.file;
      img.alt = chart.title;
      img.loading = "eager";
      frame.append(img);

      const body = createElement("div", "chart-body");
      const text = document.createElement("div");
      const title = createElement("h3", null, chart.title);
      const caption = createElement("p", null, chart.caption);
      text.append(title, caption);

      const download = createElement("a", "download-button", "Download PNG");
      download.href = chart.file;
      download.download = chart.file.split("/").pop();
      download.setAttribute("aria-label", `Download ${chart.title} as PNG`);

      body.append(text, download);
      card.append(frame, body);
      grid.append(card);
    });

    section.append(header, grid);
    root.append(section);
  });
}

renderCharts();
