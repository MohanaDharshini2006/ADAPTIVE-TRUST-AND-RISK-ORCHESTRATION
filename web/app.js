const $ = (id) => document.getElementById(id);
let policy = null;
const orders = {};

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* keep status text */ }
    throw new Error(msg);
  }
  return res.json();
}

function fillList(el, items, emptyText) {
  el.replaceChildren();
  (items.length ? items : [emptyText]).forEach((t) => {
    const li = document.createElement("li");
    li.textContent = t;                       // textContent: model output is never parsed as HTML
    el.appendChild(li);
  });
}

function showOrder() {
  const o = orders[$("order").value];
  $("orderInfo").textContent = o ? `${o.product_name}, ${o.color}, size ${o.size}, Rs ${Math.round(o.price)}, ordered ${o.order_date}` : "";
}

async function init() {
  try {
    const [list, pol, st] = await Promise.all([api("/api/orders"), api("/api/policy"), api("/api/status")]);
    policy = pol;
    list.forEach((o) => {
      orders[o.order_id] = o;
      const opt = document.createElement("option");
      opt.value = o.order_id;
      opt.textContent = `${o.order_id}: ${o.product_name}, ${o.color}`;
      $("order").appendChild(opt);
    });
    showOrder();
    $("status").textContent = `Groq key ${st.groq_key ? "set" : "not set"} | reason model ${st.reason_model || "keyword fallback"} | vision model ${st.vision_model || "not set"}`;
  } catch (e) {
    $("error").textContent = `Could not reach the API: ${e.message}`;
  }
}

function drawMatch(score) {
  const mm = policy.mismatch_threshold, hi = policy.match_threshold;
  const segs = document.querySelectorAll(".seg");
  [mm, hi - mm, 1 - hi].forEach((w, i) => { segs[i].style.width = `${w * 100}%`; });
  const labels = $("rulerLabels");
  labels.replaceChildren();
  [["Mismatch", mm], ["Verify", hi - mm], ["Match", 1 - hi]].forEach(([t, w]) => {
    const s = document.createElement("span");
    s.style.width = `${w * 100}%`;
    s.textContent = t;
    labels.appendChild(s);
  });
  const marker = $("marker");
  marker.hidden = score === null;
  if (score !== null) {
    marker.style.left = `${Math.min(Math.max(score, 0), 1) * 100}%`;
    $("markerLabel").textContent = score.toFixed(2);
  }
}

function render(data, file, orderId) {
  const { result, atro, billing } = data;
  $("verdict").className = `verdict ${atro.decision.toLowerCase()}`;
  $("decision").textContent = atro.decision;
  $("nextAction").textContent = atro.next_action.replaceAll("_", " ").toLowerCase() + (atro.conflict ? ". The claim conflicts with the evidence." : "");
  $("imgOrig").src = `/api/orders/${encodeURIComponent(orderId)}/image?t=${Date.now()}`;
  $("imgRet").src = URL.createObjectURL(file);
  drawMatch(result.product_match_score);
  $("fCondition").textContent = result.condition;
  $("fStatus").textContent = result.status.replaceAll("_", " ");
  $("fRefund").textContent = `Rs ${Math.round(result.refund_amount)}`;
  $("fTrust").textContent = atro.trust.toFixed(2);
  $("fRisk").textContent = atro.risk.toFixed(2);
  fillList($("evidence"), result.evidence, "None");
  fillList($("flags"), result.risk_flags.map((f) => f.replaceAll("_", " ")), "None");
  $("billing").textContent = billing ? JSON.stringify(billing, null, 2) : "No refund record. The return was not accepted for refund.";
  $("raw").textContent = JSON.stringify(data, null, 2);
  $("empty").hidden = true;
  $("result").hidden = false;
}

$("order").addEventListener("change", showOrder);
document.querySelectorAll("input[name=cmode]").forEach((r) =>
  r.addEventListener("change", () => { $("manual").disabled = document.querySelector("input[name=cmode]:checked").value !== "manual"; }));

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("photo").files[0];
  if (!file) { $("error").textContent = "Choose a photo of the returned item."; return; }
  const mode = document.querySelector("input[name=cmode]:checked").value;
  const body = new FormData();
  body.append("order_id", $("order").value);
  body.append("message", $("message").value);
  body.append("condition_mode", mode);
  body.append("manual_condition", $("manual").value);
  body.append("photo", file);
  $("error").textContent = "";
  $("go").disabled = true; $("go").textContent = "Checking...";
  try {
    render(await api("/api/return", { method: "POST", body }), file, $("order").value);
  } catch (err) {
    $("error").textContent = err.message;
  } finally {
    $("go").disabled = false; $("go").textContent = "Check return";
  }
});

init();
