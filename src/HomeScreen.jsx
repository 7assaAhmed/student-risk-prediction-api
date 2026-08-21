import React, { useState } from "react";
import { TrendingUp, BookOpen, Sparkles, MessageCircle, X, Search, Loader2, Send } from "lucide-react";

const API_BASE = "http://localhost:5000";

export default function HomeScreen() {
  const [studentId, setStudentId] = useState("220072");
  const [studentName, setStudentName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [courseData, setCourseData] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);

  async function loadStudent() {
    setLoading(true);
    setError(null);
    setData(null);
    setCourseData(null);
    try {
      const lookupRes = await fetch(`${API_BASE}/student/${encodeURIComponent(studentId)}`);
      if (lookupRes.status === 404) throw new Error(`No stored data for student ${studentId}.`);
      if (!lookupRes.ok) throw new Error(`Lookup failed (HTTP ${lookupRes.status})`);
      const record = await lookupRes.json();

      const payload = { student_id: studentId };
      ["year1", "year2", "year3"].forEach((y) => { if (record[y]) payload[y] = record[y]; });

      const [predictRes, coursesRes] = await Promise.all([
        fetch(`${API_BASE}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }),
        fetch(`${API_BASE}/student/${encodeURIComponent(studentId)}/courses`),
      ]);

      if (!predictRes.ok) throw new Error(`Prediction failed (HTTP ${predictRes.status})`);
      const predictData = await predictRes.json();
      setData({ ...record, predictions: predictData.predictions });

      if (coursesRes.ok) {
        setCourseData(await coursesRes.json());
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const preds = data?.predictions;

  // Real historical GPA trend (year1/2/3 actual averages) - no invented monthly points.
  const historyPoints = [];
  if (data) {
    ["year1", "year2", "year3"].forEach((y, idx) => {
      const yr = data[y];
      if (yr && (yr.fall_gpa !== undefined || yr.spring_gpa !== undefined)) {
        const vals = [yr.fall_gpa, yr.spring_gpa].filter((v) => v !== undefined && v !== null);
        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
        historyPoints.push({ label: `Year ${idx + 1}`, value: avg });
      }
    });
  }
  const currentGpa = historyPoints.length ? historyPoints[historyPoints.length - 1].value : null;
  const coursesTaken = courseData?.course_history?.length ?? null;
  const topCourseRisk = courseData?.upcoming_course_risks?.sort((a, b) => b.historical_fail_rate - a.historical_fail_rate)[0];

  const maxGpa = 4.5, chartW = 320, chartH = 130, pad = 24;

  return (
    <div className="min-h-screen bg-gray-50 relative" style={{ fontFamily: "system-ui, -apple-system, sans-serif" }}>
      {/* Header */}
      <div className="bg-gradient-to-br from-orange-400 to-amber-500 px-5 pt-6 pb-6 rounded-b-3xl">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h1 className="text-white text-2xl font-bold">Hello{studentName ? `, ${studentName}` : ""} 👋</h1>
            <p className="text-orange-50 text-sm mt-0.5">Ready to improve today?</p>
          </div>
          <button
            onClick={() => setChatOpen(true)}
            className="w-11 h-11 rounded-full bg-white/25 flex items-center justify-center text-white active:scale-95 transition"
            aria-label="Open AI Academic Advisor"
          >
            <MessageCircle size={20} />
          </button>
        </div>

        <div className="flex items-center bg-white/90 rounded-xl px-3 mb-4">
          <Search size={16} className="text-gray-400" />
          <input
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            placeholder="Student ID"
            className="flex-1 bg-transparent outline-none px-2 py-2 text-sm text-gray-800"
          />
          <button onClick={loadStudent} disabled={loading} className="text-orange-600 text-sm font-semibold px-2 flex items-center gap-1">
            {loading ? <Loader2 size={14} className="animate-spin" /> : null}
            Load
          </button>
        </div>

        {data && (
          <div className="grid grid-cols-2 gap-3">
            <StatCard icon={<TrendingUp size={20} />} value={currentGpa !== null ? currentGpa.toFixed(2) : "—"} label="Current GPA" />
            <StatCard icon={<BookOpen size={20} />} value={coursesTaken !== null ? coursesTaken : "—"} label="Courses Taken" />
          </div>
        )}
      </div>

      <div className="p-4 space-y-4 max-w-md mx-auto">
        {error && <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl p-3">{error}</div>}
        {!data && !error && !loading && (
          <div className="text-center text-gray-400 text-sm py-10">Enter a student ID and press Load.</div>
        )}

        {data && (
          <>
            {/* AI Insights - built only from real prediction fields */}
            {(preds?.predicted_next_year_gpa?.available || topCourseRisk) && (
              <div className="bg-orange-50 border border-orange-100 rounded-2xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-8 rounded-full bg-orange-400 flex items-center justify-center text-white">
                    <Sparkles size={15} />
                  </div>
                  <h3 className="font-bold text-gray-800 text-sm">AI Insights</h3>
                </div>
                <p className="text-sm text-gray-600 leading-relaxed">
                  {preds?.predicted_next_year_gpa?.available && (
                    <>Projected Year {preds.predicted_next_year_gpa.target_year} GPA: <b>{preds.predicted_next_year_gpa.value.toFixed(2)}</b>. </>
                  )}
                  {topCourseRisk && (
                    <>Based on your grade in {topCourseRisk.prerequisite_course}, past students with a similar grade failed {topCourseRisk.course_code} {(topCourseRisk.historical_fail_rate * 100).toFixed(0)}% of the time - worth extra attention.</>
                  )}
                </p>
              </div>
            )}

            {/* Performance Overview - real year-by-year GPA, not invented monthly points */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-4">
              <h3 className="font-bold text-gray-800 text-sm mb-1">Performance Overview</h3>
              <p className="text-xs text-gray-400 mb-2">Actual GPA by year on record</p>
              {historyPoints.length >= 2 ? (
                <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-32">
                  {historyPoints.map((pt, i) => {
                    if (i === 0) return null;
                    const prev = historyPoints[i - 1];
                    const x1 = pad + ((i - 1) / (historyPoints.length - 1)) * (chartW - 2 * pad);
                    const x2 = pad + (i / (historyPoints.length - 1)) * (chartW - 2 * pad);
                    const y1 = chartH - pad - (prev.value / maxGpa) * (chartH - 2 * pad);
                    const y2 = chartH - pad - (pt.value / maxGpa) * (chartH - 2 * pad);
                    return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#f97316" strokeWidth="2.5" />;
                  })}
                  {historyPoints.map((pt, i) => {
                    const x = pad + (i / (historyPoints.length - 1)) * (chartW - 2 * pad);
                    const y = chartH - pad - (pt.value / maxGpa) * (chartH - 2 * pad);
                    return (
                      <g key={i}>
                        <circle cx={x} cy={y} r="4" fill="#f97316" />
                        <text x={x} y={chartH - 4} fontSize="9" textAnchor="middle" fill="#888">{pt.label}</text>
                        <text x={x} y={y - 8} fontSize="9" textAnchor="middle" fill="#555">{pt.value.toFixed(2)}</text>
                      </g>
                    );
                  })}
                </svg>
              ) : (
                <p className="text-xs text-gray-400 py-6 text-center">Not enough recorded years to draw a trend yet.</p>
              )}
            </div>

            {/* Attendance is deliberately absent: not extracted from historical
                records - it's designed to come from real-time app usage, so
                showing a number here would be invented. */}
            <p className="text-[11px] text-gray-400 px-1">
              Attendance tracking isn't available yet - it will populate once students start checking in through the app.
            </p>
          </>
        )}
      </div>

      {chatOpen && (
        <ChatPanel studentId={studentId} onClose={() => setChatOpen(false)} />
      )}
    </div>
  );
}

function StatCard({ icon, value, label }) {
  return (
    <div className="bg-white/20 rounded-2xl p-3 text-white text-center">
      <div className="flex justify-center mb-1">{icon}</div>
      <div className="text-xl font-bold">{value}</div>
      <div className="text-[11px] text-orange-50">{label}</div>
    </div>
  );
}

function ChatPanel({ studentId, onClose }) {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi! I'm your academic advisor. Ask me about any course, or how to improve your performance." },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState(null);

  async function send() {
    const text = input.trim();
    if (!text) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setSending(true);
    setChatError(null);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: studentId,
          message: text,
          history: messages.map(({ role, content }) => ({ role, content })),
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      setMessages((m) => [...m, { role: "assistant", content: json.reply }]);
    } catch (e) {
      setChatError(e.message);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-end justify-center z-50">
      <div className="bg-white w-full max-w-md rounded-t-3xl flex flex-col" style={{ height: "80vh" }}>
        <div className="flex items-center justify-between p-4 border-b border-gray-100">
          <h2 className="font-bold text-gray-800">AI Academic Advisor</h2>
          <button onClick={onClose} className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((m, i) => (
            <div key={i} className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
              m.role === "user" ? "bg-orange-500 text-white ml-auto rounded-br-sm" : "bg-gray-100 text-gray-800 rounded-bl-sm"
            }`}>
              {m.content}
            </div>
          ))}
          {sending && <div className="text-xs text-gray-400">Thinking…</div>}
          {chatError && <div className="text-xs text-red-500">{chatError}</div>}
        </div>

        <div className="p-3 border-t border-gray-100 flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask a question…"
            className="flex-1 bg-gray-100 rounded-full px-4 py-2.5 text-sm outline-none"
          />
          <button onClick={send} disabled={sending} className="w-10 h-10 rounded-full bg-orange-500 text-white flex items-center justify-center shrink-0">
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
