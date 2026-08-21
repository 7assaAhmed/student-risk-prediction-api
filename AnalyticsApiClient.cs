// C# client for the Smart Educational Analytics API.
// Covers all 4 endpoints: /health, /student/{id}, /predict, /chat.
//
// Register in Program.cs:
//   builder.Services.AddHttpClient<AnalyticsApiClient>(c =>
//       c.BaseAddress = new Uri("http://localhost:5000")); // or your deployed URL
//
// Then inject AnalyticsApiClient wherever needed (a controller, a service, etc.)

using System.Net.Http.Json;
using System.Text.Json.Serialization;

// ============================================================
// Shared: one year's academic stats
// ============================================================
public class YearStats
{
    [JsonPropertyName("fall_gpa")] public double? FallGpa { get; set; }
    [JsonPropertyName("spring_gpa")] public double? SpringGpa { get; set; }
    [JsonPropertyName("total_courses")] public int? TotalCourses { get; set; }
    [JsonPropertyName("passed_courses")] public int? PassedCourses { get; set; }
    [JsonPropertyName("failed_courses")] public int? FailedCourses { get; set; }
    [JsonPropertyName("credits")] public double? Credits { get; set; }
    [JsonPropertyName("points")] public double? Points { get; set; }
    [JsonPropertyName("avg_grade")] public double? AvgGrade { get; set; }
    [JsonPropertyName("max_grade")] public double? MaxGrade { get; set; }
    [JsonPropertyName("min_grade")] public double? MinGrade { get; set; }
}

// ============================================================
// GET /student/{id}
// ============================================================
public class StudentLookupResponse
{
    [JsonPropertyName("student_id")] public string StudentId { get; set; } = "";
    [JsonPropertyName("year1")] public YearStats? Year1 { get; set; }
    [JsonPropertyName("year2")] public YearStats? Year2 { get; set; }
    [JsonPropertyName("year3")] public YearStats? Year3 { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
}

// ============================================================
// POST /predict
// ============================================================
public class PredictRequest
{
    [JsonPropertyName("student_id")] public string StudentId { get; set; } = "";
    [JsonPropertyName("year1")] public YearStats? Year1 { get; set; }
    [JsonPropertyName("year2")] public YearStats? Year2 { get; set; }
    [JsonPropertyName("year3")] public YearStats? Year3 { get; set; }
}

public class ExplanationFactor
{
    [JsonPropertyName("factor")] public string Factor { get; set; } = "";
    [JsonPropertyName("student_value")] public double StudentValue { get; set; }
    [JsonPropertyName("effect")] public string Effect { get; set; } = "";  // increases_risk / decreases_risk / pushes_prediction_up/down / increases_graduation_probability / decreases_graduation_probability
    [JsonPropertyName("impact")] public double Impact { get; set; }
}

// Shared shape for warning_risk / not_promoted_risk / delayed_progression_risk / graduation_probability
public class RiskPrediction
{
    [JsonPropertyName("available")] public bool Available { get; set; }
    [JsonPropertyName("probability")] public double? Probability { get; set; }
    [JsonPropertyName("predicts")] public string? Predicts { get; set; }
    [JsonPropertyName("target_year")] public int? TargetYear { get; set; }
    [JsonPropertyName("based_on")] public string? BasedOn { get; set; }
    [JsonPropertyName("confidence_note")] public string? ConfidenceNote { get; set; }
    [JsonPropertyName("explanation")] public List<ExplanationFactor>? Explanation { get; set; }
    [JsonPropertyName("reason")] public string? Reason { get; set; }  // populated when Available == false
}

public class GpaPrediction
{
    [JsonPropertyName("available")] public bool Available { get; set; }
    [JsonPropertyName("value")] public double? Value { get; set; }
    [JsonPropertyName("predicts")] public string? Predicts { get; set; }
    [JsonPropertyName("target_year")] public int? TargetYear { get; set; }
    [JsonPropertyName("confidence_note")] public string? ConfidenceNote { get; set; }
    [JsonPropertyName("explanation")] public List<ExplanationFactor>? Explanation { get; set; }
    [JsonPropertyName("reason")] public string? Reason { get; set; }
}

public class PeerCluster
{
    [JsonPropertyName("available")] public bool Available { get; set; }
    [JsonPropertyName("cluster_id")] public int? ClusterId { get; set; }
    [JsonPropertyName("description")] public string? Description { get; set; }
    [JsonPropertyName("confidence_note")] public string? ConfidenceNote { get; set; }
    [JsonPropertyName("reason")] public string? Reason { get; set; }
}

public class Predictions
{
    [JsonPropertyName("warning_risk")] public RiskPrediction? WarningRisk { get; set; }
    [JsonPropertyName("not_promoted_risk")] public RiskPrediction? NotPromotedRisk { get; set; }
    [JsonPropertyName("delayed_progression_risk")] public RiskPrediction? DelayedProgressionRisk { get; set; }
    [JsonPropertyName("predicted_next_year_gpa")] public GpaPrediction? PredictedNextYearGpa { get; set; }
    [JsonPropertyName("graduation_probability")] public RiskPrediction? GraduationProbability { get; set; }
    [JsonPropertyName("peer_cluster")] public PeerCluster? PeerCluster { get; set; }
}

public class PredictResponse
{
    [JsonPropertyName("student_id")] public string StudentId { get; set; } = "";
    [JsonPropertyName("years_provided")] public List<string> YearsProvided { get; set; } = new();
    [JsonPropertyName("predictions")] public Predictions Predictions { get; set; } = new();
}

// ============================================================
// POST /chat
// ============================================================
public class ChatMessage
{
    [JsonPropertyName("role")] public string Role { get; set; } = "";       // "user" or "assistant"
    [JsonPropertyName("content")] public string Content { get; set; } = "";
}

public class ChatRequest
{
    [JsonPropertyName("gpa")] public double? Gpa { get; set; }
    [JsonPropertyName("student_id")] public string? StudentId { get; set; }
    [JsonPropertyName("history")] public List<ChatMessage>? History { get; set; }
    [JsonPropertyName("message")] public string Message { get; set; } = "";
}

public class ChatResponse
{
    [JsonPropertyName("reply")] public string Reply { get; set; } = "";
    [JsonPropertyName("level")] public string? Level { get; set; }
    [JsonPropertyName("risk_context_used")] public bool RiskContextUsed { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
}

// ============================================================
// The client
// ============================================================
public class AnalyticsApiClient
{
    private readonly HttpClient _http;

    public AnalyticsApiClient(HttpClient http)
    {
        _http = http;
    }

    public async Task<bool> IsHealthyAsync()
    {
        var response = await _http.GetAsync("/health");
        return response.IsSuccessStatusCode;
    }

    public async Task<StudentLookupResponse?> GetStudentAsync(string studentId)
    {
        var response = await _http.GetAsync($"/student/{Uri.EscapeDataString(studentId)}");
        // 404 is a normal, expected outcome (student not in our records) - not an exception.
        return await response.Content.ReadFromJsonAsync<StudentLookupResponse>();
    }

    public async Task<PredictResponse?> PredictAsync(PredictRequest request)
    {
        var response = await _http.PostAsJsonAsync("/predict", request);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<PredictResponse>();
    }

    public async Task<ChatResponse?> ChatAsync(ChatRequest request)
    {
        var response = await _http.PostAsJsonAsync("/chat", request);
        // Don't EnsureSuccessStatusCode here - a 500 with a clear "error"
        // field (e.g. API key not configured) is still valid, readable JSON.
        return await response.Content.ReadFromJsonAsync<ChatResponse>();
    }
}

// ============================================================
// Example usage inside a controller
// ============================================================
//
// [ApiController]
// [Route("api/[controller]")]
// public class StudentRiskController : ControllerBase
// {
//     private readonly AnalyticsApiClient _analytics;
//     public StudentRiskController(AnalyticsApiClient analytics) => _analytics = analytics;
//
//     [HttpGet("{studentId}/risk")]
//     public async Task<IActionResult> GetRisk(string studentId)
//     {
//         // Option A: use our stored data for this student
//         var lookup = await _analytics.GetStudentAsync(studentId);
//         if (lookup?.Year1 == null) return NotFound("No stored data for this student.");
//
//         var result = await _analytics.PredictAsync(new PredictRequest
//         {
//             StudentId = studentId,
//             Year1 = lookup.Year1,
//             Year2 = lookup.Year2,
//             Year3 = lookup.Year3,
//         });
//         return Ok(result);
//     }
//
//     [HttpPost("{studentId}/ask")]
//     public async Task<IActionResult> Ask(string studentId, [FromBody] string question)
//     {
//         var result = await _analytics.ChatAsync(new ChatRequest
//         {
//             StudentId = studentId,
//             Message = question,
//         });
//         return Ok(result);
//     }
// }
