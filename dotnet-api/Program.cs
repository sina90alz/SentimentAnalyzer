using System.Diagnostics;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapPost("/analyze-sentiment", async context =>
{
    // Read request body as JSON
    var body = await new StreamReader(context.Request.Body).ReadToEndAsync();
    var input = JsonSerializer.Deserialize<InputText>(body);

    // Absolute paths
    var projectRoot = @"C:\Users\sina9\Documents\Python\SentimentAnalyzer";
    var scriptPath = Path.Combine(projectRoot, "ml-model", "predict.py");

    var psi = new ProcessStartInfo
    {
        FileName = @"C:\Program Files\Python313\python.exe",
        Arguments = $"\"{scriptPath}\" \"{input?.Text}\"",
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
        WorkingDirectory = Path.Combine(projectRoot, "ml-model")
    };

    var process = Process.Start(psi);
    string result = process.StandardOutput.ReadToEnd().Trim();
    string error = process.StandardError.ReadToEnd().Trim();
    process.WaitForExit();

    context.Response.ContentType = "application/json";
    await context.Response.WriteAsync(JsonSerializer.Serialize(
        new { sentiment = result, error }
    ));
});

app.Run();

public record InputText(string Text);
